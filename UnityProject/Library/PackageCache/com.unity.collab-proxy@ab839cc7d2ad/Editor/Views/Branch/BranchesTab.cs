using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.Client.Common.EventTracking;
using Codice.CM.Common;
using Codice.CM.Common.Mount;
using GluonGui;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.CodeReview;
using PlasticGui.WorkspaceWindow.QueryViews;
using PlasticGui.WorkspaceWindow.QueryViews.Branches;
using PlasticGui.WorkspaceWindow.Update;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.Views.Branches.Dialogs;

using GluonIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.IncomingChangesUpdater;
using Unity.PlasticSCM.Editor.Views.Diff;
using IGluonUpdateReport = PlasticGui.Gluon.IUpdateReport;

namespace Unity.PlasticSCM.Editor.Views.Branches
{
    internal partial class BranchesTab :
        IRefreshableView,
        IQueryRefreshableView,
        IBranchMenuOperations,
        ILaunchCodeReviewWindow,
        IGetQueryText,
        IGetFilter,
        FillBranchesView.IShowContentView,
        FillBranchesView.IShowHiddenBranchesButton
    {
        internal string EmptyStateMessage { get { return mEmptyStatePanel.Text; } }
        internal bool ShowHiddenBranchesForTesting { set { mShowHiddenBranches = value; } }
        internal DateFilter DateFilterForTesting { set { mDateFilter = value; } }
        internal IBranchMenuOperations Operations { get { return this; } }
        internal BranchesListView Table { get { return mBranchesListView; } }

        internal BranchesTab(
            WorkspaceInfo wkInfo,
            ViewHost viewHost,
            WorkspaceWindow workspaceWindow,
            IViewSwitcher viewSwitcher,
            IMergeViewLauncher mergeViewLauncher,
            IHistoryViewLauncher historyViewLauncher,
            IUpdateReport updateReport,
            IGluonUpdateReport gluonUpdateReport,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IAssetStatusCache assetStatusCache,
            ISaveAssets saveAssets,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            WorkspaceOperationsMonitor workspaceOperationsMonitor,
            IPendingChangesUpdater pendingChangesUpdater,
            IncomingChangesUpdater developerIncomingChangesUpdater,
            GluonIncomingChangesUpdater gluonIncomingChangesUpdater,
            EditorWindow parentWindow,
            bool isGluonMode,
            bool showHiddenBranches)
        {
            mWkInfo = wkInfo;
            mViewHost = viewHost;
            mWorkspaceWindow = workspaceWindow;
            mGluonUpdateReport = gluonUpdateReport;
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mAssetStatusCache = assetStatusCache;
            mSaveAssets = saveAssets;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
            mWorkspaceOperationsMonitor = workspaceOperationsMonitor;
            mPendingChangesUpdater = pendingChangesUpdater;
            mDeveloperIncomingChangesUpdater = developerIncomingChangesUpdater;
            mGluonIncomingChangesUpdater = gluonIncomingChangesUpdater;
            mParentWindow = parentWindow;
            mIsGluonMode = isGluonMode;
            mShowHiddenBranches = showHiddenBranches;

            mProgressControls = new ProgressControlsForViews();
            mShelvePendingChangesQuestionerBuilder =
                new ShelvePendingChangesQuestionerBuilder(parentWindow);
            mEnableSwitchAndShelveFeatureDialog = new EnableSwitchAndShelveFeature(
                PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                mParentWindow);
            mEmptyStatePanel = new EmptyStatePanel(parentWindow.Repaint);

            mFillBranchesView = new FillBranchesView(
                wkInfo,
                null,
                null,
                this,
                this,
                this,
                this);

            BuildComponents(
                wkInfo,
                workspaceWindow,
                workspaceWindow,
                viewSwitcher,
                historyViewLauncher,
                pendingChangesUpdater,
                developerIncomingChangesUpdater,
                gluonIncomingChangesUpdater,
                parentWindow,
                mFillBranchesView);

            mSplitterState = PlasticSplitterGUILayout.InitSplitterState(
                new float[] { 0.50f, 0.50f },
                new int[] { 100, (int)UnityConstants.BROWSE_REPOSITORY_PANEL_MIN_WIDTH },
                new int[] { 100000, 100000 }
            );

            mBranchOperations = new BranchOperations(
                wkInfo,
                workspaceWindow,
                mergeViewLauncher,
                this,
                ViewType.BranchesView,
                mProgressControls,
                updateReport,
                null,
                new ContinueWithPendingChangesQuestionerBuilder(viewSwitcher, parentWindow),
                mShelvePendingChangesQuestionerBuilder,
                new ApplyShelveWithConflictsQuestionerBuilder(),
                pendingChangesUpdater,
                developerIncomingChangesUpdater,
                shelvedChangesUpdater,
                mEnableSwitchAndShelveFeatureDialog);

            ((IRefreshableView)this).Refresh();
        }

        internal void OnEnable()
        {
            mDiffPanel.OnEnable();

            mSearchField.downOrUpArrowKeyPressed +=
                SearchField_OnDownOrUpArrowKeyPressed;
        }

        internal void OnDisable()
        {
            mDiffPanel.OnDisable();

            mSearchField.downOrUpArrowKeyPressed -=
                SearchField_OnDownOrUpArrowKeyPressed;

            TreeHeaderSettings.Save(
                mBranchesListView.multiColumnHeader.state,
                UnityConstants.BRANCHES_TABLE_SETTINGS_NAME);
        }

        internal void Update()
        {
            mDiffPanel.Update();

            mProgressControls.UpdateProgress(mParentWindow);
        }

        internal SerializableBranchesTabState GetSerializableState()
        {
            return new SerializableBranchesTabState(mShowHiddenBranches);
        }

        internal void OnGUI()
        {
            DoActionsToolbar(mProgressControls);

            PlasticSplitterGUILayout.BeginHorizontalSplit(mSplitterState);

            DoBranchesArea(
                mBranchesListView,
                mEmptyStatePanel,
                mProgressControls.IsOperationRunning());

            EditorGUILayout.BeginHorizontal();

            Rect border = GUILayoutUtility.GetRect(1, 0, 1, 100000);
            EditorGUI.DrawRect(border, UnityStyles.Colors.BarBorder);

            DoChangesArea(mDiffPanel);

            EditorGUILayout.EndHorizontal();

            PlasticSplitterGUILayout.EndHorizontalSplit();
        }

        internal void DrawSearchFieldForTab()
        {
            DrawSearchField.For(
                mSearchField,
                mBranchesListView,
                UnityConstants.SEARCH_FIELD_WIDTH);
        }

        internal void DrawDateFilter()
        {
            GUI.enabled = !mProgressControls.IsOperationRunning();

            EditorGUI.BeginChangeCheck();

            mDateFilter.FilterType = (DateFilter.Type)
                EditorGUILayout.EnumPopup(
                    mDateFilter.FilterType,
                    EditorStyles.toolbarDropDown,
                    GUILayout.Width(100));

            if (EditorGUI.EndChangeCheck())
            {
                EnumPopupSetting<DateFilter.Type>.Save(
                    mDateFilter.FilterType,
                    UnityConstants.BRANCHES_DATE_FILTER_SETTING_NAME);

                ((IRefreshableView)this).Refresh();
            }

            GUI.enabled = true;
        }

        internal void DrawShowHiddenBranchesButton()
        {
            GUI.enabled = !mProgressControls.IsOperationRunning();

            EditorGUI.BeginChangeCheck();

            mShowHiddenBranches = GUILayout.Toggle(
                mShowHiddenBranches,
                new GUIContent(
                    mShowHiddenBranches ?
                        Images.GetUnhideIcon() :
                        Images.GetHideIcon(),
                    mShowHiddenBranches ?
                        PlasticLocalization.Name.DontShowHiddenBranchesTooltip.GetString() :
                        PlasticLocalization.Name.ShowHiddenBranchesTooltip.GetString()),
                EditorStyles.toolbarButton,
                GUILayout.Width(26));

            if (EditorGUI.EndChangeCheck())
            {
                TrackFeatureUseEvent.For(
                    BranchesSelection.GetSelectedRepository(mBranchesListView),
                    TrackFeatureUseEvent.Features.Branches.ToggleShowHiddenBranches);

                ((IRefreshableView)this).Refresh();
            }

            GUI.enabled = true;
        }

        internal void SetWorkingObjectInfo(BranchInfo branchInfo)
        {
            mFillBranchesView.UpdateWorkingObject(branchInfo, mWkInfo);
        }

        internal void SetLaunchToolForTesting(
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor)
        {
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
        }

        void IRefreshableView.Refresh()
        {
            // VCS-1005209 - There are scenarios where the list of branches need to check for incoming changes.
            // For example, deleting the active branch will automatically switch your workspace to the parent changeset,
            // which might have incoming changes.
            if (mDeveloperIncomingChangesUpdater != null)
                mDeveloperIncomingChangesUpdater.Update(DateTime.Now);

            if (mGluonIncomingChangesUpdater != null)
                mGluonIncomingChangesUpdater.Update(DateTime.Now);

            RefreshAndSelect(null);
        }

        //IQueryRefreshableView
        public void RefreshAndSelect(RepObjectInfo objectToSelect)
        {
            List<IPlasticTreeNode> branchesToSelect = objectToSelect == null ?
                null : new List<IPlasticTreeNode> { new BranchTreeNode(
                    null, (BranchInfo)objectToSelect, null) };

            mDiffPanel.ClearInfo();

            mFillBranchesView.FillView(
                mBranchesListView,
                mProgressControls,
                null,
                null,
                null,
                branchesToSelect,
                FillBranchesView.ViewMode.List,
                mShowHiddenBranches);
        }

        BranchInfo IBranchMenuOperations.GetSelectedBranch()
        {
            return BranchesSelection.GetSelectedBranch(mBranchesListView);
        }

        int IBranchMenuOperations.GetSelectedBranchesCount()
        {
            return BranchesSelection.GetSelectedBranchesCount(mBranchesListView);
        }

        bool IBranchMenuOperations.AreHiddenBranchesShown()
        {
            return mShowHiddenBranches;
        }

        void IBranchMenuOperations.CreateBranch()
        {
            CreateBranchForMode();
        }

        void IBranchMenuOperations.CreateTopLevelBranch() { }

        void IBranchMenuOperations.SwitchToBranch()
        {
            SwitchToBranchForMode();
        }

        void IBranchMenuOperations.MergeBranch()
        {
            mBranchOperations.MergeBranch(
                BranchesSelection.GetSelectedRepository(mBranchesListView),
                BranchesSelection.GetSelectedBranch(mBranchesListView));
        }

        void IBranchMenuOperations.CherrypickBranch() { }

        void IBranchMenuOperations.MergeToBranch() { }

        void IBranchMenuOperations.PullBranch() { }

        void IBranchMenuOperations.PullRemoteBranch() { }

        void IBranchMenuOperations.SyncWithGit() { }

        void IBranchMenuOperations.PushBranch() { }

        void IBranchMenuOperations.DiffBranch()
        {
            LaunchDiffOperations.DiffBranch(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                BranchesSelection.GetSelectedRepository(mBranchesListView),
                BranchesSelection.GetSelectedBranch(mBranchesListView),
                mIsGluonMode);
        }

        void IBranchMenuOperations.DiffWithAnotherBranch() { }

        void IBranchMenuOperations.ViewChangesets() { }

        void IBranchMenuOperations.RenameBranch()
        {
            RepositorySpec repSpec = BranchesSelection.GetSelectedRepository(mBranchesListView);
            BranchInfo branchInfo = BranchesSelection.GetSelectedBranch(mBranchesListView);

            BranchRenameData branchRenameData = RenameBranchDialog.GetBranchRenameData(
                repSpec,
                branchInfo,
                mParentWindow);

            mBranchOperations.RenameBranch(branchRenameData);
        }

        void IBranchMenuOperations.HideUnhideBranch()
        {
            if (mShowHiddenBranches)
            {
                mBranchOperations.UnhideBranch(
                    BranchesSelection.GetSelectedRepositories(mBranchesListView),
                    BranchesSelection.GetSelectedBranches(mBranchesListView));
                return;
            }

            mBranchOperations.HideBranch(
                BranchesSelection.GetSelectedRepositories(mBranchesListView),
                BranchesSelection.GetSelectedBranches(mBranchesListView));
        }

        void IBranchMenuOperations.DeleteBranch()
        {
            var branchesToDelete = BranchesSelection.GetSelectedBranches(mBranchesListView);

            if (!DeleteBranchDialog.ConfirmDelete(mParentWindow, branchesToDelete))
                return;

            mBranchOperations.DeleteBranch(
                BranchesSelection.GetSelectedRepositories(mBranchesListView),
                branchesToDelete,
                DeleteBranchOptions.IncludeChangesets,
                !mShowHiddenBranches);
        }

        void IBranchMenuOperations.CreateCodeReview()
        {
            RepositorySpec repSpec = BranchesSelection.GetSelectedRepository(mBranchesListView);
            BranchInfo branchInfo = BranchesSelection.GetSelectedBranch(mBranchesListView);

            NewCodeReviewBehavior choice = SelectNewCodeReviewBehavior.For(repSpec.Server);

            switch (choice)
            {
                case NewCodeReviewBehavior.CreateAndOpenInDesktop:
                    mBranchOperations.CreateCodeReview(repSpec, branchInfo, this);
                    break;
                case NewCodeReviewBehavior.RequestFromUnityCloud:
                    OpenRequestReviewPage.ForBranch(repSpec, branchInfo.BranchId);
                    break;
                case NewCodeReviewBehavior.Ask:
                default:
                    break;
            }
        }

        void ILaunchCodeReviewWindow.Show(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            ReviewInfo reviewInfo,
            RepObjectInfo repObjectInfo,
            bool bShowReviewChangesTab)
        {
            LaunchTool.OpenCodeReview(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                repSpec,
                reviewInfo.Id,
                mIsGluonMode);
        }

        void IBranchMenuOperations.ViewPermissions() { }

        string IGetQueryText.Get()
        {
            return QueryConstants.BuildBranchesQuery(
                mDateFilter.GetLayoutFilter(), mShowHiddenBranches);
        }

        Filter IGetFilter.Get()
        {
            return new Filter(mBranchesListView.searchString);
        }

        void IGetFilter.Clear()
        {
            // Not used by the Plugin, needed for the Reset filters button
        }

        void FillBranchesView.IShowContentView.ShowContentPanel()
        {
            mEmptyStatePanel.UpdateContent(string.Empty);
        }

        void FillBranchesView.IShowContentView.ShowEmptyStatePanel(
            string explanationText, bool showResetFilterButton)
        {
            mEmptyStatePanel.UpdateContent(explanationText);
        }

        bool FillBranchesView.IShowHiddenBranchesButton.IsChecked
        {
            get { return mShowHiddenBranches; }
            set { mShowHiddenBranches = value; }
        }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mBranchesListView.SetFocusAndEnsureSelectedItem();
        }

        void OnSelectionChanged()
        {
            List<RepObjectInfo> selectedBranches = BranchesSelection.
                GetSelectedRepObjectInfos(mBranchesListView);

            if (selectedBranches.Count != 1)
                return;

            mDiffPanel.UpdateInfo(
                MountPointWithPath.BuildWorkspaceRootMountPoint(
                    BranchesSelection.GetSelectedRepository(mBranchesListView)),
                selectedBranches[0]);
        }

        static void DoActionsToolbar(ProgressControlsForViews progressControls)
        {
            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            if (progressControls.IsOperationRunning())
            {
                DrawProgressForViews.ForIndeterminateProgressBar(
                    progressControls.ProgressData);
            }

            GUILayout.FlexibleSpace();

            EditorGUILayout.EndHorizontal();
        }

        static void DoBranchesArea(
            BranchesListView branchesListView,
            EmptyStatePanel emptyStatePanel,
            bool isOperationRunning)
        {
            EditorGUILayout.BeginVertical();

            GUI.enabled = !isOperationRunning;

            Rect rect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

            branchesListView.OnGUI(rect);

            if (!emptyStatePanel.IsEmpty())
                emptyStatePanel.OnGUI(rect);

            GUI.enabled = true;

            EditorGUILayout.EndVertical();
        }

        static void DoChangesArea(DiffPanel diffPanel)
        {
            EditorGUILayout.BeginVertical();

            diffPanel.OnGUI();

            EditorGUILayout.EndVertical();
        }

        void BuildComponents(
            WorkspaceInfo wkInfo,
            IWorkspaceWindow workspaceWindow,
            IRefreshView refreshView,
            IViewSwitcher viewSwitcher,
            IHistoryViewLauncher historyViewLauncher,
            IPendingChangesUpdater pendingChangesUpdater,
            IncomingChangesUpdater developerIncomingChangesUpdater,
            GluonIncomingChangesUpdater gluonIncomingChangesUpdater,
            EditorWindow parentWindow,
            FillBranchesView fillBranchesView)
        {
            mSearchField = new SearchField();
            mSearchField.downOrUpArrowKeyPressed += SearchField_OnDownOrUpArrowKeyPressed;

            DateFilter.Type dateFilterType =
                EnumPopupSetting<DateFilter.Type>.Load(
                    UnityConstants.BRANCHES_DATE_FILTER_SETTING_NAME,
                    DateFilter.Type.LastMonth);
            mDateFilter = new DateFilter(dateFilterType);

            BranchesListHeaderState headerState =
                BranchesListHeaderState.GetDefault();

            TreeHeaderSettings.Load(headerState,
                UnityConstants.BRANCHES_TABLE_SETTINGS_NAME,
                (int)BranchesListColumn.CreationDate, false);

            mBranchesListView = new BranchesListView(
                headerState,
                BranchesListHeaderState.GetColumnNames(),
                new BranchesViewMenu(this, mGluonIncomingChangesUpdater != null),
                fillBranchesView,
                fillBranchesView,
                selectionChangedAction: OnSelectionChanged,
                doubleClickAction: ((IBranchMenuOperations)this).DiffBranch,
                afterItemsChangedAction: fillBranchesView.ShowContentOrEmptyState);
            mBranchesListView.Reload();

            mDiffPanel = new DiffPanel(
                wkInfo,
                workspaceWindow,
                viewSwitcher,
                historyViewLauncher,
                refreshView,
                mAssetStatusCache,
                mShowDownloadPlasticExeWindow,
                pendingChangesUpdater,
                developerIncomingChangesUpdater,
                gluonIncomingChangesUpdater,
                parentWindow,
                mIsGluonMode);
        }

        bool mShowHiddenBranches;

        DateFilter mDateFilter;
        SearchField mSearchField;
        BranchesListView mBranchesListView;
        DiffPanel mDiffPanel;

        LaunchTool.IProcessExecutor mProcessExecutor;
        LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;

        readonly object mSplitterState;

        readonly BranchOperations mBranchOperations;
        readonly FillBranchesView mFillBranchesView;
        readonly EmptyStatePanel mEmptyStatePanel;
        readonly SwitchAndShelve.IEnableSwitchAndShelveFeatureDialog mEnableSwitchAndShelveFeatureDialog;
        readonly IShelvePendingChangesQuestionerBuilder mShelvePendingChangesQuestionerBuilder;
        readonly ProgressControlsForViews mProgressControls;
        readonly bool mIsGluonMode;
        readonly EditorWindow mParentWindow;
        readonly IPendingChangesUpdater mPendingChangesUpdater;
        readonly GluonIncomingChangesUpdater mGluonIncomingChangesUpdater;
        readonly IncomingChangesUpdater mDeveloperIncomingChangesUpdater;
        readonly WorkspaceOperationsMonitor mWorkspaceOperationsMonitor;
        readonly ISaveAssets mSaveAssets;
        readonly IAssetStatusCache mAssetStatusCache;
        readonly IShelvedChangesUpdater mShelvedChangesUpdater;
        readonly IGluonUpdateReport mGluonUpdateReport;
        readonly WorkspaceWindow mWorkspaceWindow;
        readonly ViewHost mViewHost;
        readonly WorkspaceInfo mWkInfo;
    }
}
