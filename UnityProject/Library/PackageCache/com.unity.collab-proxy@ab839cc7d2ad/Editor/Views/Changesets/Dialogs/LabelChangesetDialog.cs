using UnityEditor;
using UnityEngine;

using Codice.CM.Common;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.QueryViews.Changesets;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;

namespace Unity.PlasticSCM.Editor.Views.Changesets.Dialogs
{
    class LabelChangesetDialog : PlasticDialog
    {
        protected override Rect DefaultRect
        {
            get
            {
                var baseRect = base.DefaultRect;
                return new Rect(baseRect.x, baseRect.y, 710, 290);
            }
        }

        protected override string GetTitle()
        {
            return PlasticLocalization.Name.CreateLabelTitle.GetString();
        }

        protected override void OnModalGUI()
        {
            DoTitleArea();

            DoFieldsArea();

            DoButtonsArea();
        }

        internal static ChangesetLabelData Label(
            EditorWindow parentWindow,
            RepositorySpec repSpec,
            ChangesetExtendedInfo changesetInfo)
        {
            BranchInfo parentBranchInfo = BranchInfoCache.GetBranch(
                repSpec, changesetInfo.BranchId);
            string explanation = ChangesetLabelUserInfo.GetExplanation(
                repSpec, parentBranchInfo.BranchName, changesetInfo);

            LabelChangesetDialog dialog = Create(
                repSpec, changesetInfo, explanation);

            ResponseType dialogueResult = dialog.RunModal(parentWindow);
            ChangesetLabelData result = dialog.BuildCreationData();
            result.Result = dialogueResult == ResponseType.Ok;
            return result;
        }

        void DoTitleArea()
        {
            GUILayout.BeginVertical();

            Title(PlasticLocalization.Name.CreateLabelTitle.GetString());

            GUILayout.Space(5);

            Paragraph(mExplanation);

            GUILayout.EndVertical();
        }

        void DoFieldsArea()
        {
            GUILayout.BeginVertical();

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Label(
                    PlasticLocalization.Name.LabelNameEntry.GetString(),
                    GUILayout.Width(100));

                GUI.SetNextControlName(NAME_FIELD_CONTROL_NAME);
                mLabelName = GUILayout.TextField(mLabelName);

                if (!mWasNameFieldFocused)
                {
                    EditorGUI.FocusTextInControl(NAME_FIELD_CONTROL_NAME);
                    mWasNameFieldFocused = true;
                }

                GUILayout.Space(5);
            }

            GUILayout.Space(5);

            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUILayout.VerticalScope(GUILayout.Width(100)))
                {
                    GUILayout.Space(49);
                    GUILayout.Label(
                        PlasticLocalization.Name.CommentsEntry.GetString(),
                        GUILayout.Width(100));
                }
                GUI.SetNextControlName(COMMENT_TEXTAREA_CONTROL_NAME);
                mComment = GUILayout.TextArea(mComment, GUILayout.Height(100));
                GUILayout.Space(5);
            }

            GUILayout.Space(5);

            mLabelAllXlinkedRepositories = GUILayout.Toggle(mLabelAllXlinkedRepositories,
                PlasticLocalization.Name.LabelAllXlinksCheckButton.GetString());

            GUILayout.Space(5);

            GUILayout.EndVertical();
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUILayout.HorizontalScope(GUILayout.MinWidth(500)))
                {
                    GUILayout.Space(2);
                    DrawProgressForDialogs.For(
                        mProgressControls.ProgressData);
                    GUILayout.Space(2);
                }

                GUILayout.FlexibleSpace();

                DoLabelButton();
                DoCancelButton();
            }
        }

        void DoCancelButton()
        {
            if (!NormalButton(PlasticLocalization.Name.CancelButton.GetString()))
                return;

            CancelButtonAction();
        }

        void DoLabelButton()
        {
            if (!NormalButton(PlasticLocalization.Name.LabelButton.GetString()))
                return;

            LabelButtonAction();
        }

        void LabelButtonAction()
        {
            ChangesetLabelValidation.AsyncValidation(
                BuildCreationData(), this, mProgressControls);
        }

        static LabelChangesetDialog Create(
            RepositorySpec repSpec, ChangesetExtendedInfo changesetInfo, string explanation)
        {
            var instance = CreateInstance<LabelChangesetDialog>();
            instance.IsResizable = false;
            instance.mEnterKeyAction = instance.LabelButtonAction;
            instance.AddControlConsumingEnterKey(COMMENT_TEXTAREA_CONTROL_NAME);
            instance.mEscapeKeyAction = instance.CloseButtonAction;
            instance.mRepositorySpec = repSpec;
            instance.mLabelName = "";
            instance.mComment = "";
            instance.mLabelAllXlinkedRepositories = false;
            instance.mProgressControls = new ProgressControlsForDialogs();
            instance.mExplanation = explanation;
            instance.mChangesetId = changesetInfo.ChangesetId;
            return instance;
        }

        ChangesetLabelData BuildCreationData()
        {
            mResultData = new ChangesetLabelData(
                mRepositorySpec,
                mLabelName,
                mChangesetId,
                mComment,
                mLabelAllXlinkedRepositories,
                (mResultData != null) ? mResultData.XlinksToLabel : null);

            return mResultData;
        }

        ProgressControlsForDialogs mProgressControls;
        ChangesetLabelData mResultData = null;

        RepositorySpec mRepositorySpec;
        long mChangesetId;

        string mLabelName;
        string mComment;
        bool mLabelAllXlinkedRepositories;
        string mExplanation;

        bool mWasNameFieldFocused;
        const string NAME_FIELD_CONTROL_NAME = "LabelChangesetNameField";
        const string COMMENT_TEXTAREA_CONTROL_NAME = "LabelChangesetCommentTextArea";
    }
}
