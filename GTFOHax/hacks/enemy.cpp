#include "enemy.h"
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <format>
#include <limits>
#include <thread>
#include <mutex>
#include <helpers.h>
#include "aimbot.h"
#include "esp.h"
#include "utils/math.h"

namespace Enemy
{
    std::atomic<std::shared_ptr<EnemyVec>> enemies;
    std::atomic<std::shared_ptr<EnemyVec>> enemiesReady;
    std::map<std::string, int> enemyIDs;
    std::vector<std::string> enemyNames;

    std::unordered_map<app::EnemyAgent*, EnemyPositionHistory> enemyPositionHistory;

    struct BoneLineCastCache
    {
        app::Vector3 lastBonePos      = {0.0f, 0.0f, 0.0f};
        app::Vector3 lastPlayerEyePos = {0.0f, 0.0f, 0.0f};
        bool         lastVisible      = false;
        bool         valid            = false;
    };

    struct EnemyCacheEntry
    {
        BoneLineCastCache         boneCache[64];
        app::Transform*           boneTransforms[64];
        bool                      boneTransformCached[64];
        app::Dam_EnemyDamageLimb* boneLimb[64];
        bool                      boneLimbResolved[64];
        BoneLineCastCache         fallbackCache;
        std::string               enemyName;
        uint32_t                  lastTouchFrame;

        struct CachedLimb {
            app::Dam_EnemyDamageLimb* ptr       = nullptr;
            app::Transform*           transform  = nullptr;
        };
        CachedLimb cachedLimbs[32];
        int        cachedLimbCount  = 0;
        bool       cachedLimbsReady = false;

        EnemyCacheEntry()
        {
            memset(boneTransforms, 0, sizeof(boneTransforms));
            memset(boneTransformCached, 0, sizeof(boneTransformCached));
            memset(boneLimb, 0, sizeof(boneLimb));
            memset(boneLimbResolved, 0, sizeof(boneLimbResolved));
            memset(cachedLimbs, 0, sizeof(cachedLimbs));
            lastTouchFrame = 0;
        }
    };

    static std::unordered_map<app::EnemyAgent*, EnemyCacheEntry> linecastCache;

    struct HighlightRenderer
    {
        app::Renderer* renderer = nullptr;
        int subMeshCount = 0;
    };

    struct HighlightCacheEntry
    {
        app::GameObject* root = nullptr;
        std::vector<HighlightRenderer> renderers;
    };

    struct HighlightTarget
    {
        app::EnemyAgent* enemy = nullptr;
        app::GameObject* root = nullptr;
        bool visible = false;
    };

    struct HighlightMaterialState
    {
        ImVec4 color;
        bool occludedOnly = false;
        bool valid = false;
    };

    static std::unordered_map<app::EnemyAgent*, HighlightCacheEntry> g_highlightCache;
    static std::vector<HighlightTarget> g_highlightTargets;
    static std::vector<HighlightTarget> g_highlightScratchTargets;

    struct GameVisCache
    {
        BoneLineCastCache bones[64];
        BoneLineCastCache fallback;
        BoneLineCastCache damageableBones[32];
    };
    static std::unordered_map<app::EnemyAgent*, GameVisCache> g_gameVisCache;
    struct EnemyInfoPool
    {
        std::vector<std::unique_ptr<EnemyInfo>> slots;
        std::vector<EnemyInfo*>                 freeList;
        std::mutex                              mtx;

        std::shared_ptr<EnemyInfo> acquire()
        {
            EnemyInfo* ptr;
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (!freeList.empty())
                {
                    ptr = freeList.back();
                    freeList.pop_back();
                }
                else
                {
                    slots.push_back(std::make_unique<EnemyInfo>());
                    ptr = slots.back().get();
                }
            }
            ptr->reset();
            return std::shared_ptr<EnemyInfo>(ptr, [this](EnemyInfo* p) {
                std::lock_guard<std::mutex> lock(mtx);
                freeList.push_back(p);
            });
        }
    };
    static EnemyInfoPool g_enemyInfoPool;

    static std::thread        g_refreshThread;
    static std::atomic<bool>  g_refreshRunning{false};
    static std::mutex         g_positionHistoryMtx;

    static int      g_maskDefault    = 0;
    static bool     g_maskResolved   = false;
    static uint32_t g_refreshFrame   = 0;

    static app::CommandBuffer* g_highlightBuffer = nullptr;
    static app::Material* g_visibleHighlightMaterial = nullptr;
    static app::Material* g_hiddenHighlightMaterial = nullptr;
    static app::Camera* g_highlightCamera = nullptr;
    static uint32_t g_highlightBufferHandle = 0;
    static uint32_t g_visibleHighlightMaterialHandle = 0;
    static uint32_t g_hiddenHighlightMaterialHandle = 0;
    static uint32_t g_highlightCameraHandle = 0;
    static int g_highlightColorId = -1;
    static int g_highlightSrcBlendId = -1;
    static int g_highlightDstBlendId = -1;
    static int g_highlightCullId = -1;
    static int g_highlightZWriteId = -1;
    static int g_highlightZTestId = -1;
    static HighlightMaterialState g_visibleHighlightState;
    static HighlightMaterialState g_hiddenHighlightState;
    static int32_t g_highlightReadyFrame = 0;
    static uint32_t g_highlightCacheFrame = 0;

    // Lazy cleanup: only drop a cache entry after we haven't touched it for this many
    // successive refreshes. At the 60 Hz throttle below, 600 ≈ 10 seconds.
    static constexpr uint32_t CACHE_STALE_FRAMES   = 600;
    static constexpr uint32_t CACHE_SWEEP_INTERVAL = 120;

    static app::String* ManagedString(const char* value)
    {
        return reinterpret_cast<app::String*>(il2cpp_string_new(value));
    }

    static bool IsUnityObjectAlive(const void* object)
    {
        return object &&
            app::Object_1_op_Implicit(
                reinterpret_cast<app::Object_1*>(const_cast<void*>(object)), nullptr);
    }

    static void ConfigureHighlightMaterialBase(app::Material* material)
    {
        app::Material_SetInt_1(material, g_highlightSrcBlendId, 5, nullptr);
        app::Material_SetInt_1(material, g_highlightDstBlendId, 10, nullptr);
        app::Material_SetInt_1(material, g_highlightCullId, 2, nullptr);
        app::Material_SetInt_1(material, g_highlightZWriteId, 0, nullptr);
        app::Material_set_renderQueue(material, 5000, nullptr);
    }

    static bool SameColor(const ImVec4& lhs, const ImVec4& rhs)
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
    }

    static void UpdateHighlightMaterial(
        app::Material* material,
        const ImVec4& color,
        bool occludedOnly,
        HighlightMaterialState& state)
    {
        if (state.valid && state.occludedOnly == occludedOnly && SameColor(state.color, color))
            return;

        app::Color nativeColor = { color.x, color.y, color.z, color.w };
        app::Material_SetColor_1(material, g_highlightColorId, nativeColor, nullptr);
        app::Material_SetInt_1(material, g_highlightZTestId,
            static_cast<int>(occludedOnly ? app::CompareFunction__Enum::Greater : app::CompareFunction__Enum::Always), nullptr);
        state.color = color;
        state.occludedOnly = occludedOnly;
        state.valid = true;
    }

    static void DetachHighlightCamera()
    {
        if (IsUnityObjectAlive(g_highlightCamera) && g_highlightBuffer)
            app::Camera_RemoveCommandBuffer(
                g_highlightCamera,
                app::CameraEvent__Enum::BeforeImageEffects,
                g_highlightBuffer,
                nullptr);
        if (g_highlightCameraHandle)
            il2cpp_gchandle_free(g_highlightCameraHandle);

        g_highlightCamera = nullptr;
        g_highlightCameraHandle = 0;
    }

    static bool EnsureHighlightResources()
    {
        auto currentCamera = app::Camera_get_main(nullptr);
        if (!IsUnityObjectAlive(currentCamera))
            return false;
        G::mainCamera = currentCamera;

        if (!g_highlightBuffer)
        {
            auto shader = app::Shader_Find(ManagedString("Hidden/Internal-Colored"), nullptr);
            if (!IsUnityObjectAlive(shader) ||
                !app::CommandBuffer__TypeInfo || !*app::CommandBuffer__TypeInfo ||
                !app::Material__TypeInfo || !*app::Material__TypeInfo)
                return false;

            g_highlightBuffer = reinterpret_cast<app::CommandBuffer*>(
                il2cpp_object_new(reinterpret_cast<Il2CppClass*>(*app::CommandBuffer__TypeInfo)));
            g_visibleHighlightMaterial = reinterpret_cast<app::Material*>(
                il2cpp_object_new(reinterpret_cast<Il2CppClass*>(*app::Material__TypeInfo)));
            g_hiddenHighlightMaterial = reinterpret_cast<app::Material*>(
                il2cpp_object_new(reinterpret_cast<Il2CppClass*>(*app::Material__TypeInfo)));
            if (!g_highlightBuffer || !g_visibleHighlightMaterial || !g_hiddenHighlightMaterial)
                return false;

            app::CommandBuffer__ctor(g_highlightBuffer, nullptr);
            app::CommandBuffer_set_name(g_highlightBuffer, ManagedString("GTFOHax Enemy Model Highlight"), nullptr);
            app::Material__ctor(g_visibleHighlightMaterial, shader, nullptr);
            app::Material__ctor(g_hiddenHighlightMaterial, shader, nullptr);

            g_highlightColorId = app::Shader_PropertyToID(ManagedString("_Color"), nullptr);
            g_highlightSrcBlendId = app::Shader_PropertyToID(ManagedString("_SrcBlend"), nullptr);
            g_highlightDstBlendId = app::Shader_PropertyToID(ManagedString("_DstBlend"), nullptr);
            g_highlightCullId = app::Shader_PropertyToID(ManagedString("_Cull"), nullptr);
            g_highlightZWriteId = app::Shader_PropertyToID(ManagedString("_ZWrite"), nullptr);
            g_highlightZTestId = app::Shader_PropertyToID(ManagedString("_ZTest"), nullptr);
            ConfigureHighlightMaterialBase(g_visibleHighlightMaterial);
            ConfigureHighlightMaterialBase(g_hiddenHighlightMaterial);

            g_highlightBufferHandle = il2cpp_gchandle_new(reinterpret_cast<Il2CppObject*>(g_highlightBuffer), false);
            g_visibleHighlightMaterialHandle = il2cpp_gchandle_new(reinterpret_cast<Il2CppObject*>(g_visibleHighlightMaterial), false);
            g_hiddenHighlightMaterialHandle = il2cpp_gchandle_new(reinterpret_cast<Il2CppObject*>(g_hiddenHighlightMaterial), false);
        }

        if (g_highlightCamera != currentCamera)
        {
            DetachHighlightCamera();
            g_highlightCamera = currentCamera;
            g_highlightCameraHandle =
                il2cpp_gchandle_new(reinterpret_cast<Il2CppObject*>(g_highlightCamera), false);
            app::Camera_AddCommandBuffer(g_highlightCamera, app::CameraEvent__Enum::BeforeImageEffects, g_highlightBuffer, nullptr);
        }
        return true;
    }

    static void ClearEnemyHighlights()
    {
        if (g_highlightBuffer && !g_highlightTargets.empty())
            app::CommandBuffer_Clear(g_highlightBuffer, nullptr);
        g_highlightTargets.clear();
    }

    static const std::vector<HighlightRenderer>& GetHighlightRenderers(
        app::EnemyAgent* enemy, HighlightCacheEntry& cache)
    {
        auto root = enemy->fields.MainModelGO;
        bool cacheValid = cache.root == root && !cache.renderers.empty();
        if (cacheValid)
        {
            for (const auto& entry : cache.renderers)
            {
                if (!IsUnityObjectAlive(entry.renderer))
                {
                    cacheValid = false;
                    break;
                }
            }
        }

        if (cacheValid && (g_highlightCacheFrame % 60) != 0)
            return cache.renderers;

        cache.root = root;
        cache.renderers.clear();
        if (!IsUnityObjectAlive(root) || !app::GameObject_get_activeInHierarchy(root, nullptr))
            return cache.renderers;

        if (!app::GameObject_GetComponentsInChildren_6__MethodInfo ||
            !*app::GameObject_GetComponentsInChildren_6__MethodInfo)
            return cache.renderers;

        auto renderers = app::GameObject_GetComponentsInChildren_6(
            root,
            *app::GameObject_GetComponentsInChildren_6__MethodInfo);
        if (!renderers)
            return cache.renderers;

        cache.renderers.reserve(renderers->max_length);
        for (il2cpp_array_size_t i = 0; i < renderers->max_length; ++i)
        {
            auto skinnedRenderer = renderers->vector[i];
            if (!IsUnityObjectAlive(skinnedRenderer))
                continue;

            auto renderer = reinterpret_cast<app::Renderer*>(skinnedRenderer);
            auto mesh = app::SkinnedMeshRenderer_get_sharedMesh(skinnedRenderer, nullptr);
            int subMeshCount = IsUnityObjectAlive(mesh) ? app::Mesh_get_subMeshCount(mesh, nullptr) : 1;
            cache.renderers.push_back({ renderer, (std::max)(subMeshCount, 1) });
        }
        return cache.renderers;
    }

    static void UpdateEnemyHighlights(const std::shared_ptr<EnemyVec>& snapshot)
    {
        if (!ESP::enemyESP.toggleKey.isToggled() || !snapshot)
        {
            ClearEnemyHighlights();
            return;
        }

        if (app::GameStateManager_get_CurrentStateName(nullptr) != app::eGameStateName__Enum::InLevel ||
            app::Time_get_frameCount(nullptr) < g_highlightReadyFrame)
        {
            ClearEnemyHighlights();
            return;
        }

        const bool visibleEnabled = ESP::enemyESP.visibleSec.show && ESP::enemyESP.visibleSec.showModelHighlight;
        const bool hiddenEnabled = ESP::enemyESP.nonVisibleSec.show && ESP::enemyESP.nonVisibleSec.showModelHighlight;
        if ((!visibleEnabled && !hiddenEnabled) || !EnsureHighlightResources())
        {
            ClearEnemyHighlights();
            return;
        }

        UpdateHighlightMaterial(g_visibleHighlightMaterial, ESP::enemyESP.visibleSec.modelHighlightColor,
            ESP::enemyESP.visibleSec.modelHighlightOccludedOnly, g_visibleHighlightState);
        UpdateHighlightMaterial(g_hiddenHighlightMaterial, ESP::enemyESP.nonVisibleSec.modelHighlightColor,
            ESP::enemyESP.nonVisibleSec.modelHighlightOccludedOnly, g_hiddenHighlightState);

        g_highlightScratchTargets.clear();
        g_highlightScratchTargets.reserve(snapshot->size());

        for (const auto& enemyInfo : *snapshot)
        {
            if (!enemyInfo || !IsUnityObjectAlive(enemyInfo->enemyAgent) || !enemyInfo->enemyAgent->fields.m_alive)
                continue;

            auto& section = enemyInfo->visible ? ESP::enemyESP.visibleSec : ESP::enemyESP.nonVisibleSec;
            if (!section.show || !section.showModelHighlight || enemyInfo->distance > section.renderDistance)
                continue;

            auto root = enemyInfo->enemyAgent->fields.MainModelGO;
            if (IsUnityObjectAlive(root) && app::GameObject_get_activeInHierarchy(root, nullptr))
                g_highlightScratchTargets.push_back({ enemyInfo->enemyAgent, root, enemyInfo->visible });
        }

        ClearEnemyHighlights();
        g_highlightTargets.swap(g_highlightScratchTargets);
        g_highlightScratchTargets.clear();
        ++g_highlightCacheFrame;

        for (const auto& target : g_highlightTargets)
        {
            if (!IsUnityObjectAlive(target.enemy) || !IsUnityObjectAlive(target.root))
                continue;

            auto& cache = g_highlightCache[target.enemy];
            auto material = target.visible ? g_visibleHighlightMaterial : g_hiddenHighlightMaterial;
            for (const auto& entry : GetHighlightRenderers(target.enemy, cache))
            {
                if (!IsUnityObjectAlive(entry.renderer))
                    continue;
                for (int subMesh = 0; subMesh < entry.subMeshCount; ++subMesh)
                    app::CommandBuffer_DrawRenderer(g_highlightBuffer, entry.renderer, material, subMesh, -1, nullptr);
            }
        }
    }

    static void ShutdownEnemyHighlights()
    {
        ClearEnemyHighlights();
        g_highlightCache.clear();
        g_highlightScratchTargets.clear();
        DetachHighlightCamera();
        if (g_highlightBuffer)
            app::CommandBuffer_Release(g_highlightBuffer, nullptr);
        if (IsUnityObjectAlive(g_visibleHighlightMaterial))
            app::Object_1_Destroy_1(reinterpret_cast<app::Object_1*>(g_visibleHighlightMaterial), nullptr);
        if (IsUnityObjectAlive(g_hiddenHighlightMaterial))
            app::Object_1_Destroy_1(reinterpret_cast<app::Object_1*>(g_hiddenHighlightMaterial), nullptr);

        if (g_highlightBufferHandle) il2cpp_gchandle_free(g_highlightBufferHandle);
        if (g_visibleHighlightMaterialHandle) il2cpp_gchandle_free(g_visibleHighlightMaterialHandle);
        if (g_hiddenHighlightMaterialHandle) il2cpp_gchandle_free(g_hiddenHighlightMaterialHandle);

        g_highlightBuffer = nullptr;
        g_visibleHighlightMaterial = nullptr;
        g_hiddenHighlightMaterial = nullptr;
        g_highlightBufferHandle = 0;
        g_visibleHighlightMaterialHandle = 0;
        g_hiddenHighlightMaterialHandle = 0;
        g_highlightColorId = -1;
        g_highlightSrcBlendId = -1;
        g_highlightDstBlendId = -1;
        g_highlightCullId = -1;
        g_highlightZWriteId = -1;
        g_highlightZTestId = -1;
        g_visibleHighlightState = {};
        g_hiddenHighlightState = {};
        g_highlightReadyFrame = 0;
        g_highlightCacheFrame = 0;
    }


    app::Vector3 GetEnemyMovementDirection(app::EnemyAgent* enemy)
    {
        app::Vector3 zeroVec = {0.0f, 0.0f, 0.0f};
        if (enemy == nullptr) return zeroVec;
        std::lock_guard<std::mutex> lock(g_positionHistoryMtx);
        auto it = enemyPositionHistory.find(enemy);
        if (it != enemyPositionHistory.end() && it->second.hasValidDirection)
            return it->second.movementDirection;
        return zeroVec;
    }

    static bool isBoneVisible_cached(BoneLineCastCache& cache, const app::Vector3& bonePos,
                                     const app::Vector3& eyePos)
    {
        if (cache.valid &&
            bonePos.x == cache.lastBonePos.x && bonePos.y == cache.lastBonePos.y && bonePos.z == cache.lastBonePos.z &&
            eyePos.x  == cache.lastPlayerEyePos.x && eyePos.y == cache.lastPlayerEyePos.y && eyePos.z == cache.lastPlayerEyePos.z)
        {
            return cache.lastVisible;
        }

        if (!g_maskResolved)
        {
            g_maskDefault  = (*app::LayerManager__TypeInfo)->static_fields->MASK_DEFAULT;
            g_maskResolved = true;
        }
        cache.lastVisible = !app::Physics_Linecast_1(eyePos, bonePos, g_maskDefault, NULL);
        cache.lastBonePos      = bonePos;
        cache.lastPlayerEyePos = eyePos;
        cache.valid            = true;
        return cache.lastVisible;
    }

    bool isValidDistance(bool visible, float distance)
    {
        if (visible)
        {
            if (ESP::enemyESP.visibleSec.show && distance < ESP::enemyESP.visibleSec.renderDistance)
                return true;
            if (Aimbot::settings.toggleKey.isToggled() && distance < Aimbot::settings.maxDistance)
                return true;
        }
        else
        {
            if (ESP::enemyESP.nonVisibleSec.show && distance < ESP::enemyESP.nonVisibleSec.renderDistance)
                return true;
            if (Aimbot::settings.toggleKey.isToggled() && !Aimbot::settings.visibleOnly && distance < Aimbot::settings.maxDistance)
                return true;
        }
        return false;
    }

    void _RefreshEnemyAgents()
    {
        if (G::gameQuit)
            return;
        if (!ESP::enemyESP.toggleKey.isToggled() && !Aimbot::settings.toggleKey.isToggled())
            return;
        if (G::localPlayer == nullptr)
            return;

        static bool mapsReserved = false;
        if (!mapsReserved) {
            linecastCache.reserve(512);
            enemyPositionHistory.reserve(512);
            mapsReserved = true;
        }

        ++g_refreshFrame;

        app::Vector3 localPos = G::localPlayer->fields.m_goodPosition;
        static EnemyVec enemiesTemp;
        enemiesTemp.clear();
        if (enemiesTemp.capacity() < 64) enemiesTemp.reserve(128);

        auto courseNodesList = (*app::StaticUpdateManager__TypeInfo)->static_fields->courseNodes;
        for (int i = 0; i < courseNodesList->fields._size; i++)
        {
            auto courseNode = courseNodesList->fields._items->vector[i];
            if (!courseNode) continue;
            auto enemiesList = courseNode->fields.m_enemiesInNode;
            if (!enemiesList) continue;

            for (int j = 0; j < enemiesList->fields._size; j++)
            {
                auto enemyAgent = enemiesList->fields._items->vector[j];
                if (enemyAgent == NULL || !enemyAgent->fields.m_alive)
                    continue;

                app::Vector3 enemyPos = enemyAgent->fields._.m_position;
                float dx = enemyPos.x - localPos.x, dy = enemyPos.y - localPos.y, dz = enemyPos.z - localPos.z;
                float distanceSq = dx*dx + dy*dy + dz*dz;
                float distance = sqrtf(distanceSq);

                float maxDist = (std::max<float>)((std::max<float>)((float)ESP::enemyESP.visibleSec.renderDistance, (float)ESP::enemyESP.nonVisibleSec.renderDistance), (float)Aimbot::settings.maxDistance);
                if (distance > maxDist)
                {
                    if (distance > maxDist + 50.0f) linecastCache.erase(enemyAgent);
                    continue;
                }

                {
                    std::lock_guard<std::mutex> lock(g_positionHistoryMtx);
                    auto hit = enemyPositionHistory.find(enemyAgent);
                    if (hit != enemyPositionHistory.end())
                    {
                        hit->second.previousPosition = hit->second.currentPosition;
                        hit->second.currentPosition  = enemyPos;
                        float pdx = enemyPos.x - hit->second.previousPosition.x;
                        float pdz = enemyPos.z - hit->second.previousPosition.z;
                        float lenSq = pdx*pdx + pdz*pdz;
                        if (lenSq > 0.0001f)
                        {
                            float len = sqrtf(lenSq);
                            hit->second.movementDirection = { pdx / len, 0.0f, pdz / len };
                            hit->second.hasValidDirection = true;
                        }
                        hit->second.lastTouchFrame = g_refreshFrame;
                    }
                    else
                    {
                        enemyPositionHistory[enemyAgent] = { enemyPos, enemyPos, {0.0f, 0.0f, 0.0f}, false, g_refreshFrame };
                    }
                }

                auto& cacheEntry = linecastCache[enemyAgent];
                cacheEntry.lastTouchFrame = g_refreshFrame;
                if (cacheEntry.enemyName.empty())
                    cacheEntry.enemyName = il2cppi_to_string(
                        app::Object_1_GetName(reinterpret_cast<app::Object_1*>(enemyAgent), NULL));

                struct TempLimb { app::Vector3 pos; app::Dam_EnemyDamageLimb* ptr; };
                TempLimb tempLimbs[32];
                int tempLimbCount = 0;

                auto enemyInfo = g_enemyInfoPool.acquire();
                enemyInfo->enemyAgent = enemyAgent;
                enemyInfo->enemyObjectName = cacheEntry.enemyName;
                enemyInfo->distance = distance;

                auto damageLimbsList = enemyAgent->fields.Damage->fields.DamageLimbs;
                if (damageLimbsList) {
                    if (!cacheEntry.cachedLimbsReady) {
                        auto damageLimbs = damageLimbsList->vector;
                        int limbCount = (std::min)((int)damageLimbsList->max_length, 32);
                        for (int k = 0; k < limbCount; k++) {
                            auto limb = damageLimbs[k];
                            if (!limb) continue;
                            auto xform = app::Component_1_get_transform(
                                reinterpret_cast<app::Component_1*>(limb), NULL);
                            if (!xform) continue;
                            auto& cl = cacheEntry.cachedLimbs[cacheEntry.cachedLimbCount++];
                            cl.ptr       = limb;
                            cl.transform = xform;
                        }
                        cacheEntry.cachedLimbsReady = true;
                    }
                    for (int k = 0; k < cacheEntry.cachedLimbCount; k++) {
                        auto& cl = cacheEntry.cachedLimbs[k];
                        app::Vector3 limbPos;
                        app::Transform_get_position_Injected(cl.transform, &limbPos, NULL);
                        tempLimbs[tempLimbCount++] = { limbPos, cl.ptr };
                        Bone& bone = enemyInfo->damageableBones[enemyInfo->damageableBoneCount++];
                        bone.position = limbPos;
                        bone.damageable = true;
                        bone.destroyed = cl.ptr->fields._IsDestroyed_k__BackingField;
                        bone.limbType  = cl.ptr->fields.m_type;
                        bone.health    = cl.ptr->fields.m_health;
                        bone.limbPtr   = cl.ptr;
                    }
                }
                for (auto boneType : Enemy::WantedBones)
                {
                    int idx = static_cast<int>(boneType);
                    if (idx < 0 || idx >= 64) continue;

                    app::Transform* boneTransform;
                    if (cacheEntry.boneTransformCached[idx])
                    {
                        boneTransform = cacheEntry.boneTransforms[idx];
                    }
                    else
                    {
                        boneTransform = app::Animator_GetBoneTransform(enemyAgent->fields.Anim, boneType, NULL);
                        cacheEntry.boneTransforms[idx] = boneTransform;
                        cacheEntry.boneTransformCached[idx] = true;
                    }
                    if (boneTransform == nullptr) continue;

                    Bone bone;
                    app::Transform_get_position_Injected(boneTransform, &bone.position, NULL);

                    app::Dam_EnemyDamageLimb* matchedLimb = cacheEntry.boneLimb[idx];
                    if (matchedLimb == nullptr && !cacheEntry.boneLimbResolved[idx])
                    {
                        for (int k = 0; k < tempLimbCount; k++) {
                            if (tempLimbs[k].pos.x == bone.position.x &&
                                tempLimbs[k].pos.y == bone.position.y &&
                                tempLimbs[k].pos.z == bone.position.z) {
                                matchedLimb = tempLimbs[k].ptr;
                                cacheEntry.boneLimb[idx] = matchedLimb;
                                break;
                            }
                        }
                        cacheEntry.boneLimbResolved[idx] = true;
                    }
                    if (matchedLimb)
                    {
                        bone.damageable = true;
                        bone.destroyed  = matchedLimb->fields._IsDestroyed_k__BackingField;
                        bone.limbType   = matchedLimb->fields.m_type;
                        bone.health     = matchedLimb->fields.m_health;
                        bone.limbPtr    = matchedLimb;
                    }

                    enemyInfo->bones[idx] = std::move(bone);
                    enemyInfo->hasBone[idx] = true;
                }

                bool hasEssentialBones = enemyInfo->hasBone[static_cast<int>(app::HumanBodyBones__Enum::Head)] &&
                                         enemyInfo->hasBone[static_cast<int>(app::HumanBodyBones__Enum::LeftFoot)] &&
                                         enemyInfo->hasBone[static_cast<int>(app::HumanBodyBones__Enum::RightFoot)];

                if (!hasEssentialBones)
                {
                    enemyInfo->fallbackBone.position = enemyPos;
                    enemyInfo->useFallback = true;
                }

                // Include enemy if it would be valid under *any* visibility state.
                // Actual visible flag is set by UpdateEnemyVisibility() on the game thread.
                if (isValidDistance(true, distance) || isValidDistance(false, distance))
                    enemiesTemp.push_back(std::move(enemyInfo));
            }
        }

        if ((g_refreshFrame % CACHE_SWEEP_INTERVAL) == 0)
        {
            for (auto it = linecastCache.begin(); it != linecastCache.end(); )
            {
                if (it->second.lastTouchFrame + CACHE_STALE_FRAMES < g_refreshFrame)
                    it = linecastCache.erase(it);
                else
                    ++it;
            }
            {
                std::lock_guard<std::mutex> lock(g_positionHistoryMtx);
                for (auto it = enemyPositionHistory.begin(); it != enemyPositionHistory.end(); )
                {
                    if (it->second.lastTouchFrame + CACHE_STALE_FRAMES < g_refreshFrame)
                        it = enemyPositionHistory.erase(it);
                    else
                        ++it;
                }
            }
        }

        enemies.store(std::make_shared<EnemyVec>(std::move(enemiesTemp)));
    }

    void _SpawnEnemy(int id, app::AgentMode__Enum agentMode)
    {
        app::EnemyAllocator* enemyAllocator = (*app::EnemyAllocator__TypeInfo)->static_fields->Current;
        app::PlayerAgent* localPlayer = app::PlayerManager_2_GetLocalPlayerAgent(nullptr);
        if (!localPlayer) return;
        app::Agent* localPlayerAgent = reinterpret_cast<app::Agent*>(localPlayer);
        app::Quaternion playerRotation = app::Agent_get_Rotation(localPlayerAgent, NULL);
        app::AIG_CourseNode* courseNode = localPlayer->fields.m_courseNode;

        app::Vector3 screenCenter = { G::screenWidth / 2.0f, G::screenHeight / 2.0f, 0 };
        app::Ray screenCenterRay = app::Camera_ScreenPointToRay_2(G::mainCamera, screenCenter, NULL);
        app::RaycastHit raycastHit;
        if (app::Physics_Raycast_14(screenCenterRay, &raycastHit, 200, NULL))
        {
            app::EnemyAllocator_ResetAllowedToSpawn(NULL);
            app::EnemyAllocator_SpawnEnemy(enemyAllocator, id, courseNode, agentMode, raycastHit.m_Point, playerRotation, nullptr, 0, nullptr);
        }
    }

    void UpdateEnemyVisibility()
    {
        if (!ESP::enemyESP.toggleKey.isToggled())
            ClearEnemyHighlights();
        if (!ESP::enemyESP.toggleKey.isToggled() && !Aimbot::settings.toggleKey.isToggled())
            return;
        if (G::localPlayer == nullptr)
        {
            ClearEnemyHighlights();
            return;
        }

        const bool needVisibilityCheck =
            ESP::enemyESP.visibleSec.show || ESP::enemyESP.nonVisibleSec.show ||
            (Aimbot::settings.toggleKey.isToggled() && Aimbot::settings.visibleOnly);
        if (!needVisibilityCheck)
        {
            ClearEnemyHighlights();
            return;
        }

        static bool reservedOnce = false;
        if (!reservedOnce)
        {
            g_gameVisCache.reserve(512);
            reservedOnce = true;
        }

        app::Vector3 eyePos = G::localPlayer->fields.m_eyePosition;

        auto snapshot = enemies.load();
        if (!snapshot)
        {
            ClearEnemyHighlights();
            return;
        }

        for (auto& enemyInfo : *snapshot)
        {
            if (!enemyInfo) continue;
            app::EnemyAgent* agent = enemyInfo->enemyAgent;
            auto& visCache = g_gameVisCache[agent];

            bool anyVisible = false;

            if (enemyInfo->useFallback)
            {
                isBoneVisible_cached(visCache.fallback, enemyInfo->fallbackBone.position, eyePos);
                enemyInfo->fallbackBone.visible = visCache.fallback.lastVisible;
                anyVisible = visCache.fallback.lastVisible;
            }
            else
            {
                for (auto boneType : Enemy::WantedBones)
                {
                    int idx = static_cast<int>(boneType);
                    if (idx < 0 || idx >= 64 || !enemyInfo->hasBone[idx]) continue;
                    isBoneVisible_cached(visCache.bones[idx], enemyInfo->bones[idx].position, eyePos);
                    enemyInfo->bones[idx].visible = visCache.bones[idx].lastVisible;
                    if (visCache.bones[idx].lastVisible) anyVisible = true;
                }
            }
            for (int k = 0; k < enemyInfo->damageableBoneCount; k++)
            {
                isBoneVisible_cached(visCache.damageableBones[k], enemyInfo->damageableBones[k].position, eyePos);
                enemyInfo->damageableBones[k].visible = visCache.damageableBones[k].lastVisible;
            }
            enemyInfo->visible = anyVisible;
        }

        UpdateEnemyHighlights(snapshot);
        enemiesReady.store(snapshot);
    }

    void OnGameStateChanged(app::eGameStateName__Enum nextState)
    {
        ClearEnemyHighlights();
        g_highlightCache.clear();
        g_highlightScratchTargets.clear();
        g_highlightCacheFrame = 0;

        DetachHighlightCamera();
        G::mainCamera = nullptr;
        g_highlightReadyFrame =
            nextState == app::eGameStateName__Enum::InLevel
                ? app::Time_get_frameCount(nullptr) + 30
                : (std::numeric_limits<int32_t>::max)();
    }

    void RefreshEnemyAgents()
    {
        static std::once_flag s_startOnce;
        std::call_once(s_startOnce, []() {
            g_refreshRunning.store(true, std::memory_order_relaxed);
            g_refreshThread = std::thread([]() {
                while (g_refreshRunning.load(std::memory_order_relaxed))
                {
                    _RefreshEnemyAgents();
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                }
            });
        });
    }

    void StopRefreshThread()
    {
        g_refreshRunning.store(false, std::memory_order_relaxed);
        if (g_refreshThread.joinable())
            g_refreshThread.join();
        ShutdownEnemyHighlights();
    }

    void SpawnEnemy(int id, app::AgentMode__Enum agentMode)
    {
        G::callbacks.push([id, agentMode] { _SpawnEnemy(id, agentMode); });
    }
}
