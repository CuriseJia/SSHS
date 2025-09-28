using UnityEditor;
using UnityEngine;

using Codice.CM.Common;
using PlasticGui;
using PlasticGui.WorkspaceWindow.QueryViews.Labels;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.Views.Changesets;

namespace Unity.PlasticSCM.Editor.Views.Labels.Dialogs
{
    class CreateLabelDialog : PlasticDialog
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
            if (mShowChangesetExplorer)
            {
                // HACK: GetTitle is not called once the dialog is shown
                titleContent.text = PlasticLocalization.Name.AvailableChangesets.GetString();
                mChangesetExplorerView.OnGUI();

                return;
            }

            titleContent.text = PlasticLocalization.Name.CreateLabelTitle.GetString();
            mCreateLabelView.OnGUI();
        }

        internal static LabelCreationData CreateLabel(
            EditorWindow parentWindow,
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            MarkerExtendedInfo markerInfo)
        {
            CreateLabelDialog dialog = Create(
                parentWindow, wkInfo, repSpec, markerInfo.Changeset);

            ResponseType dialogResult = dialog.RunModal(parentWindow);
            LabelCreationData result = mCreateLabelView.BuildCreationData();
            result.Result = dialogResult == ResponseType.Ok;
            return result;
        }

        static CreateLabelDialog Create(
            EditorWindow parentWindow,
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            long changesetId)
        {
            CreateLabelDialog instance = CreateInstance<CreateLabelDialog>();
            instance.IsResizable = false;

            mCreateLabelView = new CreateLabelView(instance, repSpec);

            mChangesetExplorerView = new ChangesetExplorerView(
                instance, wkInfo, new ProgressControlsForDialogs());

            instance.mEnterKeyAction = instance.CreateButtonAction;
            instance.mEscapeKeyAction = instance.CloseButtonAction;

            instance.ToggleChangesetExplorer(false);
            instance.SetChangesetId(changesetId);

            return instance;
        }

        internal void CreateButtonAction()
        {
            LabelCreationValidation.AsyncValidation(
                mCreateLabelView.BuildCreationData(), this, mCreateLabelView.ProgressControls);
        }

        internal void ToggleChangesetExplorer(bool show)
        {
            mShowChangesetExplorer = show;
        }

        internal void SetChangesetId(long changesetId)
        {
            mCreateLabelView.SetChangesetId(changesetId);
        }

        static CreateLabelView mCreateLabelView;
        static ChangesetExplorerView mChangesetExplorerView;

        static bool mShowChangesetExplorer;
    }
}
