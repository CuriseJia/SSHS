using Unity.PlasticSCM.Editor.CloudDrive;
using UnityEditor;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class CloseWindowIfOpened
    {
        internal static void UVCS()
        {
            if (!EditorWindow.HasOpenInstances<UVCSWindow>())
                return;

            UVCSWindow window = EditorWindow.
                GetWindow<UVCSWindow>(null, false);

            window.Close();
        }

        internal static void CloudDrive()
        {
            if (!EditorWindow.HasOpenInstances<CloudDriveWindow>())
                return;

            CloudDriveWindow window = EditorWindow.
                GetWindow<CloudDriveWindow>(null, false);

            window.Close();
        }
    }
}
