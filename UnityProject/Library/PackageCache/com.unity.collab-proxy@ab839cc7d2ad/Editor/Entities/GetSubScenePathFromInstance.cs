#if HAS_ENTITIES_PACKAGE
using UnityEditor;
using UnityEngine;
using Unity.Scenes;

using Unity.PlasticSCM.Editor.AssetsOverlays;

using Object = UnityEngine.Object;

namespace Unity.PlasticSCM.Editor.Entities
{
    internal class GetSubScenePathFromInstance : DrawHierarchyOverlay.IGetAssetPathFromInstance
    {
        internal static void Register()
        {
            DrawHierarchyOverlay.GetSubSceneAssetPath = new GetSubScenePathFromInstance();
        }

        bool DrawHierarchyOverlay.IGetAssetPathFromInstance.TryGetAssetPath(
            int instanceID,
            out string assetPath)
        {
            assetPath = null;

            Object hierarchyObject = EditorUtility.InstanceIDToObject(instanceID);

            if (hierarchyObject == null || hierarchyObject is not GameObject)
                return false;

            assetPath = GetSubScenePathFromGameObject((GameObject) hierarchyObject);

            return assetPath != null;
        }

        string GetSubScenePathFromGameObject(GameObject gameObject)
        {
            SubScene subSceneComponent = gameObject.GetComponent<SubScene>();

            return subSceneComponent == null ? null : subSceneComponent.EditableScenePath;
        }
    }
}
#endif
