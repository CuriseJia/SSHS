using UnityEditor;
using UnityEngine;

using Codice.CM.Common;

using PlasticGui;
using PlasticGui.WorkspaceWindow.QueryViews.Labels;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;

namespace Unity.PlasticSCM.Editor.Views.Labels.Dialogs
{
    internal class RenameLabelDialog : PlasticDialog
    {
        protected override Rect DefaultRect
        {
            get
            {
                var baseRect = base.DefaultRect;
                return new Rect(baseRect.x, baseRect.y, 500, 200);
            }
        }

        internal static LabelRenameData GetLabelRenameData(
            RepositorySpec repSpec,
            MarkerInfo labelInfo,
            EditorWindow parentWindow)
        {
            RenameLabelDialog dialog = Create(
                repSpec,
                labelInfo,
                new ProgressControlsForDialogs());

            ResponseType dialogResult = dialog.RunModal(parentWindow);

            LabelRenameData result = dialog.BuildRenameData();

            result.Result = dialogResult == ResponseType.Ok;
            return result;
        }

        static RenameLabelDialog Create(
            RepositorySpec repSpec,
            MarkerInfo labelInfo,
            ProgressControlsForDialogs progressControls)
        {
            var instance = CreateInstance<RenameLabelDialog>();
            instance.mEnterKeyAction = instance.OkButtonWithValidationAction;
            instance.mEscapeKeyAction = instance.CancelButtonAction;
            instance.mRepSpec = repSpec;
            instance.mLabelInfo = labelInfo;
            instance.mLabelName = labelInfo.Name;
            instance.mTitle = PlasticLocalization.GetString(
               PlasticLocalization.Name.RenameLabelTitle);
            instance.mProgressControls = progressControls;
            return instance;
        }

        protected override string GetTitle()
        {
            return mTitle;
        }

        protected override void OnModalGUI()
        {
            Title(mTitle);

            GUILayout.Space(10f);

            DoInputArea();

            GUILayout.Space(10f);

            DrawProgressForDialogs.For(mProgressControls.ProgressData);

            GUILayout.Space(10f);

            DoButtonsArea();
        }

        void DoInputArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Label(
                    PlasticLocalization.GetString(PlasticLocalization.Name.NewName),
                    GUILayout.ExpandWidth(false));

                GUILayout.Space(10f);

                GUI.SetNextControlName(RENAME_LABEL_TEXTAREA_NAME);

                mLabelName = GUILayout.TextField(
                    mLabelName,
                    GUILayout.ExpandWidth(true));

                if (!mTextAreaFocused)
                {
                    EditorGUI.FocusTextInControl(RENAME_LABEL_TEXTAREA_NAME);
                    mTextAreaFocused = true;
                }
            }
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.FlexibleSpace();

                if (Application.platform == RuntimePlatform.WindowsEditor)
                {
                    DoOkButton();
                    DoCancelButton();
                    return;
                }

                DoCancelButton();
                DoOkButton();
            }
        }

        void DoOkButton()
        {
            if (!NormalButton(PlasticLocalization.GetString(
                    PlasticLocalization.Name.RenameButton)))
                return;

            OkButtonWithValidationAction();
        }

        void DoCancelButton()
        {
            if (!NormalButton(PlasticLocalization.GetString(
                    PlasticLocalization.Name.CancelButton)))
                return;

            CancelButtonAction();
        }

        void OkButtonWithValidationAction()
        {
            LabelRenameValidation.AsyncValidation(
                BuildRenameData(),
                this,
                mProgressControls);
        }

        LabelRenameData BuildRenameData()
        {
            return new LabelRenameData(mRepSpec, mLabelInfo, mLabelName);
        }

        string mTitle;
        string mLabelName;

        bool mTextAreaFocused;

        RepositorySpec mRepSpec;
        MarkerInfo mLabelInfo;

        ProgressControlsForDialogs mProgressControls;

        const string RENAME_LABEL_TEXTAREA_NAME = "rename_label_textarea";
    }
}
