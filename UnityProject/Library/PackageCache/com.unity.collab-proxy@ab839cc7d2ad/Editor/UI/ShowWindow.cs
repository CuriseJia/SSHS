using System;

using UnityEditor;

using Unity.PlasticSCM.Editor.CloudDrive;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class ShowWindow
    {
        // This legacy method is kept for backward compatibility with the Build Automation package
        // (it was the only method up to com.unity.collab-proxy "2.8.2")
        // UUM-115391 "Failed to open UVCS Window" error is thrown [...] settings in Build Profiles
        internal static UVCSWindow Plastic()
        {
            return UVCS();
        }

        internal static UVCSWindow UVCS()
        {
            UVCSWindow window = EditorWindow.GetWindow<UVCSWindow>(
                UnityConstants.UVCS_WINDOW_TITLE,
                true,
                mConsoleWindowType,
                mProjectBrowserType);

            window.titleContent.image = UVCSPlugin.Instance.GetPluginStatusIcon();

            return window;
        }

        internal static CloudDriveWindow CloudDrive()
        {
            CloudDriveWindow window = EditorWindow.GetWindow<CloudDriveWindow>(
                UnityConstants.CloudDrive.WINDOW_TITLE,
                true,
                mConsoleWindowType,
                mProjectBrowserType);

            window.titleContent.image = Images.GetCloudDriveViewIcon();

            return window;
        }

        static Type mConsoleWindowType = typeof(EditorWindow).
            Assembly.GetType("UnityEditor.ConsoleWindow");
        static Type mProjectBrowserType = typeof(EditorWindow).
            Assembly.GetType("UnityEditor.ProjectBrowser");
    }
}
