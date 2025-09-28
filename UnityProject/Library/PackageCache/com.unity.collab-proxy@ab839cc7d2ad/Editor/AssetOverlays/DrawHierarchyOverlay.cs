using System;

using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

using Codice.Client.Common.Threading;
using Codice.LogWrapper;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetUtils;

namespace Unity.PlasticSCM.Editor.AssetsOverlays
{
    internal static class DrawHierarchyOverlay
    {
        internal interface IGetAssetPathFromInstance
        {
            bool TryGetAssetPath(int instanceID, out string assetPath);
        }

        internal static IGetAssetPathFromInstance GetSubSceneAssetPath;

        internal static void Enable(
            string wkPath,
            IAssetStatusCache assetStatusCache)
        {
            if (mIsEnabled)
                return;

            mLog.Debug("Enable");

            mWkPath = wkPath;
            mAssetStatusCache = assetStatusCache;

            mIsEnabled = true;

            EditorApplication.hierarchyWindowItemOnGUI += OnHierarchyGUI;

            RepaintEditor.HierarchyWindow();
        }

        internal static void Disable()
        {
            mLog.Debug("Disable");

            mIsEnabled = false;

            EditorApplication.hierarchyWindowItemOnGUI -= OnHierarchyGUI;

            RepaintEditor.HierarchyWindow();

            mWkPath = null;
            mAssetStatusCache = null;
        }

        static void OnHierarchyGUI(int instanceID, Rect selectionRect)
        {
            if (Event.current.type != EventType.Repaint)
                return;

            try
            {
                string assetPath = GetAssetPathFromInstanceID(instanceID);

                if (assetPath == null)
                    return;

                string assetFullPath = AssetsPath.GetFullPathUnderWorkspace.ForAsset(mWkPath, assetPath);

                if (assetFullPath == null)
                    return;

                DrawOverlayForAsset(assetFullPath, selectionRect, mAssetStatusCache);
            }
            catch (Exception ex)
            {
                ExceptionsHandler.LogException(typeof(DrawHierarchyOverlay).Name, ex);
            }
        }

        static string GetAssetPathFromInstanceID(int instanceID)
        {
            string sceneAssetPath;
            if (TryGetAssetPathForScene(instanceID, out sceneAssetPath))
            {
                return sceneAssetPath;
            }

            string subSceneAssetPath;
            if (TryGetAssetPathForSubScene(instanceID, out subSceneAssetPath))
            {
                return subSceneAssetPath;
            }

            return null;
        }

        static bool TryGetAssetPathForScene(int instanceID, out string assetPath)
        {
            assetPath = null;

            if (EditorUtility.InstanceIDToObject(instanceID) != null)
                return false;

            assetPath = FindScenePathForHandle(instanceID);

            return assetPath != null;
        }

        static string FindScenePathForHandle(int sceneHandle)
        {
            for (int i = 0; i < SceneManager.sceneCount; i++)
            {
                Scene scene = SceneManager.GetSceneAt(i);

                if (scene.handle == sceneHandle && scene.path != null)
                {
                    return scene.path;
                }
            }

            return null;
        }

        static bool TryGetAssetPathForSubScene(int instanceID, out string assetPath)
        {
            assetPath = null;

            string subSceneAssetPath;
            if (GetSubSceneAssetPath != null &&
                GetSubSceneAssetPath.TryGetAssetPath(instanceID, out subSceneAssetPath))
            {
                assetPath = subSceneAssetPath;
            }

            return assetPath != null;
        }

        static void DrawOverlayForAsset(
            string assetFullPath,
            Rect selectionRect,
            IAssetStatusCache assetStatusCache)
        {
            AssetStatus assetStatus = assetStatusCache.GetStatus(assetFullPath);

            string tooltipText = AssetOverlay.GetTooltipText(
                assetStatus,
                assetStatusCache.GetLockStatusData(assetFullPath));

            DrawAssetOverlayIcon.ForStatus(
                selectionRect,
                assetStatus,
                tooltipText);
        }

        static bool mIsEnabled;
        static IAssetStatusCache mAssetStatusCache;
        static string mWkPath;

        static readonly ILog mLog = PlasticApp.GetLogger("DrawHierarchyOverlay");
    }
}
