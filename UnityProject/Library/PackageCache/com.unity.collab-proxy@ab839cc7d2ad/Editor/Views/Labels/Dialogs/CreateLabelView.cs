using UnityEditor;
using UnityEngine;

using Codice.CM.Common;
using PlasticGui;
using PlasticGui.WorkspaceWindow.QueryViews.Labels;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;

namespace Unity.PlasticSCM.Editor.Views.Labels.Dialogs
{
    internal class CreateLabelView : IPlasticDialogCloser
    {
        internal ProgressControlsForDialogs ProgressControls
        {
            get { return mProgressControls; }
        }

        internal CreateLabelView(
            CreateLabelDialog parentWindow,
            RepositorySpec repSpec)
        {
            mParentWindow = parentWindow;
            mRepositorySpec = repSpec;
            mNewLabelName = "";
            mComment = "";
            mLabelAllXlinkedRepositories = false;
            mSwitchToLabel = false;
            mProgressControls = new ProgressControlsForDialogs();
        }

        internal void OnGUI()
        {
            DoTitleArea();

            DoFieldsArea();

            DoButtonsArea();

            mProgressControls.ForcedUpdateProgress(mParentWindow);
        }

        void DoTitleArea()
        {
            GUILayout.BeginVertical();

            GUILayout.Label(PlasticLocalization.Name.CreateLabelTitle.GetString(),
                UnityStyles.Dialog.Title);

            GUILayout.Space(5);

            GUILayout.Label(PlasticLocalization.Name.CreateLabelExplanation.GetString(),
                UnityStyles.Paragraph);

            GUILayout.Space(10);

            GUILayout.EndVertical();
        }

        void DoFieldsArea()
        {
            GUILayout.BeginVertical();

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Label(
                    PlasticLocalization.Name.LabelNameEntry.GetString(),
                    GUILayout.Width(120));

                GUI.SetNextControlName(NAME_FIELD_CONTROL_NAME);
                mNewLabelName = GUILayout.TextField(mNewLabelName);

                if (!mWasNameFieldFocused)
                {
                    EditorGUI.FocusTextInControl(NAME_FIELD_CONTROL_NAME);
                    mWasNameFieldFocused = true;
                }
            }

            GUILayout.Space(5);

            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUILayout.VerticalScope(GUILayout.Width(100)))
                {
                    GUILayout.Space(49);
                    GUILayout.Label(
                        PlasticLocalization.Name.CommentsEntry.GetString(),
                        GUILayout.Width(120));
                }

                GUI.SetNextControlName(COMMENT_TEXTAREA_CONTROL_NAME);
                mComment = GUILayout.TextArea(mComment, GUILayout.Height(100));
            }

            GUILayout.Space(10);

            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.Label(
                    PlasticLocalization.Name.ChangesetToLabelEntry.GetString(),
                    GUILayout.Width(120));

                GUILayout.Label(mChangesetId, GUILayout.Width(450));

                if (GUILayout.Button(
                        PlasticLocalization.Name.ChooseMessage.GetString(), EditorStyles.miniButton))
                {
                    DoChooseButton();
                }
            }

            GUILayout.Space(10);

            mLabelAllXlinkedRepositories = GUILayout.Toggle(mLabelAllXlinkedRepositories,
                PlasticLocalization.Name.LabelAllXlinksCheckButton.GetString());

            GUILayout.Space(5);

            mSwitchToLabel = GUILayout.Toggle(mSwitchToLabel,
                PlasticLocalization.Name.SwitchToLabelCheckButton.GetString());

            GUILayout.Space(5);

            GUILayout.EndVertical();
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUILayout.HorizontalScope(GUILayout.MinWidth(450)))
                {
                    GUILayout.Space(2);
                    DrawProgressForDialogs.For(
                        mProgressControls.ProgressData);
                    GUILayout.Space(2);
                }

                GUILayout.FlexibleSpace();

                DoCreateButton();
                DoCancelButton();
            }
        }

        void DoChooseButton()
        {
            mParentWindow.ToggleChangesetExplorer(true);
        }

        void DoCancelButton()
        {
            if (!GUILayout.Button(
                    PlasticLocalization.Name.CancelButton.GetString(), UnityStyles.Dialog.NormalButton,
                GUILayout.MinWidth(80),
                GUILayout.Height(25)))
                return;

            CancelButtonAction();
        }

        void DoCreateButton()
        {
            if (!GUILayout.Button(
                    PlasticLocalization.Name.CreateButton.GetString(), UnityStyles.Dialog.NormalButton,
                GUILayout.MinWidth(80),
                GUILayout.Height(25)))
                return;

            mParentWindow.CreateButtonAction();
        }

        void CancelButtonAction()
        {
            mParentWindow.CancelButtonAction();
        }

        internal LabelCreationData BuildCreationData()
        {
            mResultData = new LabelCreationData(
                mRepositorySpec,
                long.Parse(mChangesetId),
                mNewLabelName,
                mComment,
                mSwitchToLabel,
                mLabelAllXlinkedRepositories,
                (mResultData != null) ? mResultData.XlinksToLabel : null);

            return mResultData;
        }

        void IPlasticDialogCloser.CloseDialog()
        {
            mParentWindow.ToggleChangesetExplorer(false);
        }

        internal void SetChangesetId(long changesetId)
        {
            mChangesetId = changesetId.ToString();
        }

        LabelCreationData mResultData = null;

        string mNewLabelName;
        string mComment;
        string mChangesetId;

        bool mLabelAllXlinkedRepositories;
        bool mSwitchToLabel;
        bool mWasNameFieldFocused;

        readonly ProgressControlsForDialogs mProgressControls;
        readonly CreateLabelDialog mParentWindow;
        readonly RepositorySpec mRepositorySpec;

        const string NAME_FIELD_CONTROL_NAME = "CreateLabelNameField";
        const string COMMENT_TEXTAREA_CONTROL_NAME = "CreateLabelCommentTextArea";
    }
}
