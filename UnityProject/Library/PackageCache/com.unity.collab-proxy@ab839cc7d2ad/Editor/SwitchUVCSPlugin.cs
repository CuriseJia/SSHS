using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor
{
    internal static class SwitchUVCSPlugin
    {
        internal static UVCSWindow On(UVCSPlugin uvcsPlugin)
        {
            uvcsPlugin.Enable();

            UVCSWindow window = ShowWindow.UVCS();

            UVCSPluginIsEnabledPreference.Enable();

            return window;
        }

        internal static UVCSWindow OnIfNeeded(UVCSPlugin uvcsPlugin)
        {
            if (!UVCSPluginIsEnabledPreference.IsEnabled())
                return On(uvcsPlugin);

            return ShowWindow.UVCS();
        }

        internal static void Off(UVCSPlugin uvcsPlugin)
        {
            UVCSPluginIsEnabledPreference.Disable();

            CloseWindowIfOpened.UVCS();

            uvcsPlugin.Shutdown();
        }
    }
}
