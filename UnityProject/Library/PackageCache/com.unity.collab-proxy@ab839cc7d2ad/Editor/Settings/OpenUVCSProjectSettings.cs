using System;
using System.Reflection;

using UnityEditor;

using Codice.Client.Common.Threading;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Settings
{
    internal static class OpenUVCSProjectSettings
    {
        internal static void ByDefault()
        {
            UVCSProjectSettingsProvider provider = OpenInUVCSProjectSettings();

            if (provider == null)
                return;

            provider.OpenAllFoldouts();
        }

        internal static void InDiffAndMergeFoldout()
        {
            UVCSProjectSettingsProvider provider = OpenInUVCSProjectSettings();

            if (provider == null)
                return;

            provider.OpenDiffAndMergeFoldout();
        }

        internal static void InShelveAndSwitchFoldout()
        {
            UVCSProjectSettingsProvider provider = OpenInUVCSProjectSettings();

            if (provider == null)
                return;

            provider.OpenShelveAndSwitchFoldout();
        }

        internal static void InOtherFoldout()
        {
            UVCSProjectSettingsProvider provider = OpenInUVCSProjectSettings();

            if (provider == null)
                return;

            provider.OpenOtherFoldout();
        }

        internal static UVCSProjectSettingsProvider OpenInUVCSProjectSettings()
        {
            EditorWindow settingsWindow = OpenProjectSettingsWithUVCSSelected();
            return GetUVCSProvider(settingsWindow);
        }

        internal static EditorWindow OpenProjectSettingsWithUVCSSelected()
        {
            return SettingsService.OpenProjectSettings(
                UnityConstants.PROJECT_SETTINGS_TAB_PATH);
        }

        internal static UVCSProjectSettingsProvider GetUVCSProvider(
            EditorWindow settingsWindow)
        {
            try
            {
                /* The following code must be compiled only for editor versions that allow our code
                 to access internal code from the editor, otherwise the ProjectSettingsWindow is not
                 accessible and the compilation fails.
                 Unity 6000.3.0a3 */
#if UNITY_6000_3_OR_NEWER
                ProjectSettingsWindow projectSettingsWindow = settingsWindow as ProjectSettingsWindow;
                return projectSettingsWindow.GetCurrentProvider() as UVCSProjectSettingsProvider;
#else
                MethodInfo getCurrentProviderMethod = settingsWindow.GetType().GetMethod(
                    "GetCurrentProvider",
                    BindingFlags.Instance | BindingFlags.NonPublic);

                return getCurrentProviderMethod.Invoke(
                    settingsWindow, null) as UVCSProjectSettingsProvider;
#endif
            }
            catch (Exception ex)
            {
                ExceptionsHandler.LogException("OpenUVCSProjectSettings", ex);
                return null;
            }
        }
    }
}
