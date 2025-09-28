using UnityEditor;
using UnityEngine;

using PlasticGui;
using Unity.PlasticSCM.Editor;
using Unity.PlasticSCM.Editor.Configuration;

namespace Unity.Cloud.Collaborate
{
    [InitializeOnLoad]
    internal static class ToolbarBootstrap
    {
        static ToolbarBootstrap()
        {
            if (ToolConfig.EnableNewUVCSToolbarButtonTokenExists())
                return;

            ToolbarButton.Initialize(UVCSPlugin.Instance);
        }
    }

    internal class ToolbarButton : SubToolbar
    {
        internal static void Initialize(UVCSPlugin uvcsPlugin)
        {
            ToolbarButton toolbar = new ToolbarButton(uvcsPlugin);
            Toolbar.AddSubToolbar(toolbar);
        }

        ToolbarButton(UVCSPlugin uvcsPlugin)
        {
            mUVCSPlugin = uvcsPlugin;
            mButtonGUIContent = new GUIContent(
                string.Empty, PlasticLocalization.Name.UnityVersionControl.GetString());

            Width = 32f;

            mUVCSPlugin.OnNotificationStatusUpdated += OnUVCSNotificationUpdated;
        }

        ~ToolbarButton()
        {
            mUVCSPlugin.OnNotificationStatusUpdated -= OnUVCSNotificationUpdated;
        }

        void OnUVCSNotificationUpdated()
        {
            Toolbar.RepaintToolbar();
        }

        public override void OnGUI(Rect rect)
        {
            Texture icon = mUVCSPlugin.GetPluginStatusIcon();
            EditorGUIUtility.SetIconSize(new Vector2(16, 16));

            mButtonGUIContent.image = icon;

            if (GUI.Button(rect, mButtonGUIContent, "AppCommand"))
                SwitchUVCSPlugin.OnIfNeeded(mUVCSPlugin);

            EditorGUIUtility.SetIconSize(Vector2.zero);
        }

        readonly GUIContent mButtonGUIContent;
        readonly UVCSPlugin mUVCSPlugin;
    }
}
