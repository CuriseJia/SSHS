using UnityEditor;

using Unity.PlasticSCM.Editor.CloudDrive;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class GetWindowIfOpened
    {
        internal static UVCSWindow UVCS()
        {
            if (!EditorWindow.HasOpenInstances<UVCSWindow>())
                return null;

            return EditorWindow.GetWindow<UVCSWindow>(null, false);
        }

        internal static CloudDriveWindow CloudDrive()
        {
            if (!EditorWindow.HasOpenInstances<CloudDriveWindow>())
                return null;

            return EditorWindow.GetWindow<CloudDriveWindow>(null, false);
        }
    }
}
