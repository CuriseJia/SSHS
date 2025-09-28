using System;

using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

using Codice.Client.Common.EventTracking;
using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.Toolbar;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Settings
{
    internal class UVCSProjectSettingsProvider : SettingsProvider
    {
        [SettingsProvider]
        internal static SettingsProvider CreateSettingsProvider()
        {
            if (!FindWorkspace.HasWorkspace(ApplicationDataPath.Get()))
                return null;

            PlasticApp.InitializeIfNeeded();

            return new UVCSProjectSettingsProvider(
                UnityConstants.PROJECT_SETTINGS_TAB_PATH,
                SettingsScope.Project,
                UVCSPlugin.Instance,
                UVCSToolbar.Controller);
        }

        UVCSProjectSettingsProvider(
            string path,
            SettingsScope scope,
            UVCSPlugin uvcsPlugin,
            IUpdateToolbarButtonVisibility updateToolbarButtonVisibility)
            : base(path, scope)
        {
            mUVCSPlugin = uvcsPlugin;
            mUpdateToolbarButtonVisibility = updateToolbarButtonVisibility;

            label = UnityConstants.PROJECT_SETTINGS_TAB_TITLE;

            OpenAllFoldouts();
        }

        public override void OnActivate(
            string searchContext,
            VisualElement rootElement)
        {
            mIsUVCSPluginEnabled = UVCSPluginIsEnabledPreference.IsEnabled();

            mShowToolbarButton = UVCSToolbarButtonIsShownPreference.IsEnabled();

            mWkInfo = FindWorkspace.InfoForApplicationPath(
                ApplicationDataPath.Get(), PlasticGui.Plastic.API);

            mIsProjectSettingsActivated = true;

            mPendingChangesOptionsFoldout.OnActivate(mWkInfo, mUVCSPlugin.PendingChangesUpdater);
            mDiffAndMergeOptionsFoldout.OnActivate();
            mShelveAndSwitchOptionsFoldout.OnActivate();
            mOtherOptionsFoldout.OnActivate();
        }

        public override void OnDeactivate()
        {
            if (!mIsProjectSettingsActivated)
                return;

            mIsProjectSettingsActivated = false;

            mPendingChangesOptionsFoldout.OnDeactivate(mUVCSPlugin.PendingChangesUpdater);
            mDiffAndMergeOptionsFoldout.OnDeactivate();
            mShelveAndSwitchOptionsFoldout.OnDeactivate();
            mOtherOptionsFoldout.OnDeactivate();
        }

        public override void OnGUI(string searchContext)
        {
            DrawSettingsSection(DoIsEnabledSetting);

            // DrawSettingsSection(DoShowToolbarButtonSetting);

            if (!mIsUVCSPluginEnabled)
                return;

            mIsPendingChangesFoldoutOpen = DrawFoldout(
                mIsPendingChangesFoldoutOpen,
                PlasticLocalization.Name.PendingChangesOptionsSectionTitle.GetString(),
                mPendingChangesOptionsFoldout.OnGUI);

            mIsDiffAndMergeFoldoutOpen = DrawFoldout(
                mIsDiffAndMergeFoldoutOpen,
                PlasticLocalization.Name.DiffAndMergeOptionsSectionTitle.GetString(),
                mDiffAndMergeOptionsFoldout.OnGUI);

            mIsShelveAndSwitchFoldoutOpen = DrawFoldout(
                mIsShelveAndSwitchFoldoutOpen,
                PlasticLocalization.Name.ShelveAndSwitchOptionsSectionTitle.GetString(),
                mShelveAndSwitchOptionsFoldout.OnGUI);

            mIsOtherFoldoutOpen = DrawFoldout(
                mIsOtherFoldoutOpen,
                PlasticLocalization.Name.OtherOptionsSectionTitle.GetString(),
                mOtherOptionsFoldout.OnGUI);
        }

        internal void OpenAllFoldouts()
        {
            mIsPendingChangesFoldoutOpen = true;
            mIsDiffAndMergeFoldoutOpen = true;
            mIsShelveAndSwitchFoldoutOpen = true;
            mIsOtherFoldoutOpen = true;
        }

        internal void OpenDiffAndMergeFoldout()
        {
            mIsDiffAndMergeFoldoutOpen = true;
            mIsPendingChangesFoldoutOpen = false;
            mIsShelveAndSwitchFoldoutOpen = false;
            mIsOtherFoldoutOpen = false;
        }

        internal void OpenShelveAndSwitchFoldout()
        {
            mIsShelveAndSwitchFoldoutOpen = true;
            mIsPendingChangesFoldoutOpen = false;
            mIsDiffAndMergeFoldoutOpen = false;
            mIsOtherFoldoutOpen = false;
        }

        internal void OpenOtherFoldout()
        {
            mIsOtherFoldoutOpen = true;
            mIsPendingChangesFoldoutOpen = false;
            mIsDiffAndMergeFoldoutOpen = false;
            mIsShelveAndSwitchFoldoutOpen = false;
        }

        internal void ShowToolbarButtonCheckboxRaiseCheckedForTesting()
        {
            mShowToolbarButton = true;
        }

        internal void ShowToolbarButtonCheckboxRaiseUncheckedForTesting()
        {
            mShowToolbarButton = false;
        }

        void DoIsEnabledSetting()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                string message = PlasticLocalization.GetString(
                    mIsUVCSPluginEnabled ?
                        PlasticLocalization.Name.UnityVCSIsEnabled :
                        PlasticLocalization.Name.UnityVCSIsDisabled);

                GUILayout.Label(
                    message,
                    EditorStyles.boldLabel,
                    GUILayout.Height(20));

                EditorGUILayout.Space(8);

                DoIsEnabledButton();

                GUILayout.FlexibleSpace();
            }
        }

        void DoIsEnabledButton()
        {
            if (!GUILayout.Button(PlasticLocalization.GetString(
                    mIsUVCSPluginEnabled ?
                        PlasticLocalization.Name.DisableButton :
                        PlasticLocalization.Name.EnableButton),
                    UnityStyles.ProjectSettings.ToggleOn))
            {
                return;
            }

            if (!mIsUVCSPluginEnabled)
            {
                mIsUVCSPluginEnabled = true;

                TrackFeatureUseEvent.For(
                    PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                    TrackFeatureUseEvent.Features.UnityPackage.EnableManually);

                SwitchUVCSPlugin.On(mUVCSPlugin);
                return;
            }

            if (mIsUVCSPluginEnabled)
            {
                mIsUVCSPluginEnabled = false;

                TrackFeatureUseEvent.For(
                    PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                    TrackFeatureUseEvent.Features.UnityPackage.DisableManually);

                SwitchUVCSPlugin.Off(mUVCSPlugin);
                return;
            }
        }

        void DoShowToolbarButtonSetting()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                bool wasChecked = mShowToolbarButton;

                bool isChecked = EditorGUILayout.Toggle(Styles.ShowToolbarButton, wasChecked);

                if (!wasChecked && isChecked)
                {
                    mShowToolbarButton = true;
                    mUpdateToolbarButtonVisibility.Show();
                    return;
                }

                if (wasChecked && !isChecked)
                {
                    mShowToolbarButton = false;
                    mUpdateToolbarButtonVisibility.Hide();
                    return;
                }
            }
        }

        static void DrawSettingsSection(Action drawSettings)
        {
            float originalLabelWidth = EditorGUIUtility.labelWidth;

            try
            {
                EditorGUIUtility.labelWidth = UnityConstants.SETTINGS_GUI_WIDTH;

                using (new EditorGUILayout.HorizontalScope())
                {
                    GUILayout.Space(20);

                    using (new EditorGUILayout.VerticalScope())
                    {
                        GUILayout.Space(10);

                        drawSettings();

                        GUILayout.Space(10);
                    }

                    GUILayout.Space(10);
                }
            }
            finally
            {
                EditorGUIUtility.labelWidth = originalLabelWidth;
            }
        }

        static bool DrawFoldout(
            bool isFoldoutOpen,
            string title,
            Action drawContent)
        {
            EditorGUILayout.BeginVertical(EditorStyles.helpBox);

            bool result =
                EditorGUILayout.BeginFoldoutHeaderGroup(
                    isFoldoutOpen,
                    title,
                    UnityStyles.ProjectSettings.FoldoutHeader);

            if (result)
                drawContent();

            EditorGUILayout.EndFoldoutHeaderGroup();
            EditorGUILayout.EndVertical();

            return result;
        }

        class Styles
        {
            internal static GUIContent ShowToolbarButton =
                new GUIContent(
                    PlasticLocalization.Name.ShowUnityVersionControlToolbarButton.GetString());
        }

        bool mIsPendingChangesFoldoutOpen;
        bool mIsDiffAndMergeFoldoutOpen;
        bool mIsShelveAndSwitchFoldoutOpen;
        bool mIsOtherFoldoutOpen;
        bool mIsProjectSettingsActivated;
        bool mShowToolbarButton;
        bool mIsUVCSPluginEnabled;

        WorkspaceInfo mWkInfo;

        PendingChangesOptionsFoldout mPendingChangesOptionsFoldout = new PendingChangesOptionsFoldout();
        DiffAndMergeOptionsFoldout mDiffAndMergeOptionsFoldout = new DiffAndMergeOptionsFoldout();
        ShelveAndSwitchOptionsFoldout mShelveAndSwitchOptionsFoldout = new ShelveAndSwitchOptionsFoldout();
        OtherOptionsFoldout mOtherOptionsFoldout = new OtherOptionsFoldout();

        readonly IUpdateToolbarButtonVisibility mUpdateToolbarButtonVisibility;
        readonly UVCSPlugin mUVCSPlugin;
    }
}
