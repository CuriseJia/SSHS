using System.Collections.Generic;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.CM.Common;
using GluonGui;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.QueryViews;
using PlasticGui.WorkspaceWindow.QueryViews.Labels;
using PlasticGui.WorkspaceWindow.Views.QueryViews.Labels;
using PlasticGui.WorkspaceWindow.Update;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.Views.BrowseRepository;
using Unity.PlasticSCM.Editor.Views.Labels.Dialogs;

using GluonIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.IncomingChangesUpdater;
using IGluonUpdateReport = PlasticGui.Gluon.IUpdateReport;

namespace Unity.PlasticSCM.Editor.Views.Labels
{
    internal partial class LabelsTab :
        IRefreshableView,
        IQueryRefreshableView,
        ILabelMenuOperations,
        IGetQueryText,
        IGetFilterText,
        FillLabelsView.IShowContentView
    {
        internal LabelsListView Table { get { return mLabelsListView; } }
        internal ILabelMenuOperations Operations { get { return this; } }
        internal string EmptyStateMessage { get { return mEmptyStatePanel.Text; } }
        internal DateFilter DateFilterForTesting { set { mDateFilter = value; } }

        internal LabelsTab(
            WorkspaceInfo wkInfo,
            WorkspaceWindow workspaceWindow,
            IViewSwitcher viewSwitcher,
            IMergeViewLauncher mergeViewLauncher,
            ViewHost viewHost,
            IUpdateReport updateReport,
            IGluonUpdateReport gluonUpdateReport,
            IPendingChangesUpdater pendingChangesUpdater,
            IncomingChangesUpdater developerIncomingChangesUpdater,
            GluonIncomingChangesUpdater gluonIncomingChangesUpdater,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IAssetStatusCache assetStatusCache,
            LaunchTool.IShowDownloadPlasticExeWindow showDownloadPlasticExeWindow,
            LaunchTool.IProcessExecutor processExecutor,
            EditorWindow parentWindow,
            bool isGluonMode)
        {
            mWkInfo = wkInfo;
            mViewHost = viewHost;
            mWorkspaceWindow = workspaceWindow;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
            mParentWindow = parentWindow;
            mIsGluonMode = isGluonMode;
            mGluonUpdateReport = gluonUpdateReport;
            mPendingChangesUpdater = pendingChangesUpdater;
            mGluonIncomingChangesUpdater = gluonIncomingChangesUpdater;
            mShelvePendingChangesQuestionerBuilder = new ShelvePendingChangesQuestionerBuilder(parentWindow);
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mAssetStatusCache = assetStatusCache;
            mShelvePendingChangesQuestionerBuilder =
                new ShelvePendingChangesQuestionerBuilder(parentWindow);
            mEnableSwitchAndShelveFeatureDialog = new EnableSwitchAndShelveFeature(
                PlasticGui.Plastic.API.GetRepositorySpec(mWkInfo),
                mParentWindow);
            mEmptyStatePanel = new EmptyStatePanel(parentWindow.Repaint);

            mProgressControls = new ProgressControlsForViews();

            mFillLabelsView = new FillLabelsView(
                wkInfo,
                null,
                null,
                this,
                this,
                this);

            BuildComponents(
                wkInfo,
                workspaceWindow,
                viewSwitcher,
                mergeViewLauncher,
                updateReport,
                developerIncomingChangesUpdater,
                shelvedChangesUpdater,
                mShelvePendingChangesQuestionerBuilder,
                mEnableSwitchAndShelveFeatureDialog,
                parentWindow,
                mFillLabelsView);

            mSplitterState = PlasticSplitterGUILayout.InitSplitterState(
                new float[] { 0.50f, 0.50f },
                new int[] { 100, (int)UnityConstants.BROWSE_REPOSITORY_PANEL_MIN_WIDTH },
                new int[] { 100000, 100000 }
            );

            mLabelOperations = new LabelOperations(
                wkInfo,
                workspaceWindow,
                mergeViewLauncher,
                this,
                ViewType.LabelsView,
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

            RefreshAndSelect(null);
        }

        internal void OnEnable()
        {
            mBrowseRepositoryPanel.OnEnable();

            mSearchField.downOrUpArrowKeyPressed +=
                SearchField_OnDownOrUpArrowKeyPressed;
        }

        internal void OnDisable()
        {
            mBrowseRepositoryPanel.OnDisable();

            mSearchField.downOrUpArrowKeyPressed -=
                SearchField_OnDownOrUpArrowKeyPressed;

            TreeHeaderSettings.Save(
                mLabelsListView.multiColumnHeader.state,
                UnityConstants.LABELS_TABLE_SETTINGS_NAME);
        }

        internal void Update()
        {
            mBrowseRepositoryPanel.Update();

            mProgressControls.UpdateProgress(mParentWindow);
        }

        internal void OnGUI()
        {
            PlasticSplitterGUILayout.BeginHorizontalSplit(mSplitterState);

            DoLabelsArea(
                mLabelsListView,
                mEmptyStatePanel,
                mProgressControls);

            DoContentBrowserArea(
                mBrowseRepositoryPanel,
                mProgressControls.IsOperationRunning());

            PlasticSplitterGUILayout.EndHorizontalSplit();
        }

        internal void DrawSearchFieldForTab()
        {
            DrawSearchField.For(
                mSearchField,
                mLabelsListView,
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
                    UnityConstants.LABELS_DATE_FILTER_SETTING_NAME);

                ((IRefreshableView)this).Refresh();
            }

            GUI.enabled = true;
        }

        internal void RefreshAndSelect(RepObjectInfo repObj)
        {
            List<object> labelsToSelect = repObj == null ?
                null : new List<object> { repObj };

            mBrowseRepositoryPanel.ClearInfo();

            mFillLabelsView.FillView(
                mLabelsListView,
                mProgressControls,
                null,
                null,
                null,
                labelsToSelect);
        }

        void IRefreshableView.Refresh()
        {
            RefreshAndSelect(null);
        }

        void IQueryRefreshableView.RefreshAndSelect(RepObjectInfo repObj)
        {
            RefreshAndSelect(repObj);
        }

        int ILabelMenuOperations.GetSelectedLabelsCount()
        {
            return LabelsSelection.GetSelectedLabelsCount(mLabelsListView);
        }

        void ILabelMenuOperations.CreateLabel()
        {
            RepositorySpec repSpec =
                LabelsSelection.GetSelectedRepository(mLabelsListView);
            MarkerExtendedInfo label =
                LabelsSelection.GetSelectedLabel(mLabelsListView);

            LabelCreationData labelCreationData = CreateLabelDialog.CreateLabel(
                mParentWindow,
                mWkInfo,
                repSpec,
                label);

            mLabelOperations.CreateLabel(
                labelCreationData,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    mAssetStatusCache,
                    ProjectPackages.ShouldBeResolvedFromUpdateReport(mWkInfo, items)));
        }

        void ILabelMenuOperations.ApplyLabelToWorkspace()
        {
            RepositorySpec repSpec =
                LabelsSelection.GetSelectedRepository(mLabelsListView);
            MarkerExtendedInfo label =
                LabelsSelection.GetSelectedLabel(mLabelsListView);

            mLabelOperations.ApplyLabelToWorkspace(repSpec, label);
        }

        void ILabelMenuOperations.SwitchToLabel()
        {
            RepositorySpec repSpec =
                LabelsSelection.GetSelectedRepository(mLabelsListView);
            MarkerExtendedInfo label =
                LabelsSelection.GetSelectedLabel(mLabelsListView);

            mLabelOperations.SwitchToLabel(
                repSpec,
                label,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    mAssetStatusCache,
                    ProjectPackages.ShouldBeResolvedFromUpdateReport(mWkInfo, items)));
        }

        void ILabelMenuOperations.BrowseRepositoryOnLabel() { }

        void ILabelMenuOperations.DiffWithAnotherLabel() { }

        void ILabelMenuOperations.DiffSelectedLabels()
        {
            RepositorySpec repSpec =
                LabelsSelection.GetSelectedRepository(mLabelsListView);
            List<RepObjectInfo> selectedLabels =
                LabelsSelection.GetSelectedRepObjectInfos(mLabelsListView);

            if (selectedLabels.Count < 2)
                return;

            LaunchDiffOperations.DiffSelectedLabels(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                repSpec,
                (MarkerExtendedInfo)selectedLabels[0],
                (MarkerExtendedInfo)selectedLabels[1],
                mIsGluonMode);
        }

        void ILabelMenuOperations.MergeLabel()
        {
            RepositorySpec repSpec =
                LabelsSelection.GetSelectedRepository(mLabelsListView);
            MarkerExtendedInfo label =
                LabelsSelection.GetSelectedLabel(mLabelsListView);

            mLabelOperations.MergeLabel(repSpec, label);
        }

        void ILabelMenuOperations.MergeToLabel() { }

        void ILabelMenuOperations.CreateBranchFromLabel()
        {
            CreateBranchForMode();
        }

        void ILabelMenuOperations.RenameLabel()
        {
            RepositorySpec repSpec =
                LabelsSelection.GetSelectedRepository(mLabelsListView);
            MarkerExtendedInfo label =
                LabelsSelection.GetSelectedLabel(mLabelsListView);

            LabelRenameData labelRenameData = RenameLabelDialog.GetLabelRenameData(
                repSpec,
                label,
                mParentWindow);

            mLabelOperations.RenameLabel(labelRenameData);
        }

        void ILabelMenuOperations.DeleteLabel()
        {
            RepositorySpec repSpec =
                LabelsSelection.GetSelectedRepository(mLabelsListView);
            MarkerExtendedInfo label =
                LabelsSelection.GetSelectedLabel(mLabelsListView);

            mLabelOperations.DeleteLabel(
                new List<RepositorySpec>() { repSpec },
                new List<MarkerExtendedInfo>() { label });
        }

        void ILabelMenuOperations.ViewPermissions() { }

        string IGetQueryText.Get()
        {
            return GetLabelsQuery(mDateFilter);
        }

        string IGetFilterText.Get()
        {
            return mLabelsListView.searchString;
        }

        void IGetFilterText.Clear()
        {
            // Not used by the Plugin, needed for the Reset filters button
        }

        void FillLabelsView.IShowContentView.ShowContentPanel()
        {
            mEmptyStatePanel.UpdateContent(string.Empty);
        }

        void FillLabelsView.IShowContentView.ShowEmptyStatePanel(
            string explanationText, bool showResetFilterButton)
        {
            mEmptyStatePanel.UpdateContent(explanationText);
        }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mLabelsListView.SetFocusAndEnsureSelectedItem();
        }

        static string GetLabelsQuery(DateFilter dateFilter)
        {
            if (dateFilter.FilterType == DateFilter.Type.AllTime)
                return QueryConstants.LabelsBeginningQuery;

            string whereClause = QueryConstants.GetDateWhereClause(
                dateFilter.GetTimeAgo());

            return string.Format("{0} {1}",
                QueryConstants.LabelsBeginningQuery,
                whereClause);
        }

        void OnSelectionChanged()
        {
            List<RepObjectInfo> selectedLabels = LabelsSelection.
                GetSelectedRepObjectInfos(mLabelsListView);

            if (selectedLabels.Count != 1)
                return;

            mBrowseRepositoryPanel.UpdateInfo(
                (MarkerExtendedInfo)selectedLabels[0]);
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

        static void DoLabelsArea(
            LabelsListView labelsListView,
            EmptyStatePanel emptyStatePanel,
            ProgressControlsForViews progressControls)
        {
            EditorGUILayout.BeginVertical();

            DoActionsToolbar(progressControls);

            GUI.enabled = !progressControls.IsOperationRunning();

            Rect rect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

            labelsListView.OnGUI(rect);

            if (!emptyStatePanel.IsEmpty())
                emptyStatePanel.OnGUI(rect);

            GUI.enabled = true;

            EditorGUILayout.EndVertical();
        }

        static void DoContentBrowserArea(
            BrowseRepositoryPanel browseRepositoryPanel,
            bool isOperationRunning)
        {
            EditorGUILayout.BeginHorizontal();

            GUI.enabled = !isOperationRunning;

            Rect border = GUILayoutUtility.GetRect(1, 0, 1, 100000);
            EditorGUI.DrawRect(border, UnityStyles.Colors.BarBorder);

            browseRepositoryPanel.OnGUI();

            GUI.enabled = true;

            EditorGUILayout.EndHorizontal();
        }

        void BuildComponents(
            WorkspaceInfo wkInfo,
            IWorkspaceWindow workspaceWindow,
            IViewSwitcher viewSwitcher,
            IMergeViewLauncher mergeViewLauncher,
            IUpdateReport updateReport,
            IncomingChangesUpdater developerIncomingChangesUpdater,
            IShelvedChangesUpdater shelvedChangesUpdater,
            IShelvePendingChangesQuestionerBuilder shelvePendingChangesQuestionerBuilder,
            SwitchAndShelve.IEnableSwitchAndShelveFeatureDialog enableSwitchAndShelveFeatureDialog,
            EditorWindow parentWindow,
            FillLabelsView fillLabelsView)
        {
            mSearchField = new SearchField();
            mSearchField.downOrUpArrowKeyPressed += SearchField_OnDownOrUpArrowKeyPressed;

            DateFilter.Type dateFilterType =
                EnumPopupSetting<DateFilter.Type>.Load(
                    UnityConstants.LABELS_DATE_FILTER_SETTING_NAME,
                    DateFilter.Type.LastMonth);
            mDateFilter = new DateFilter(dateFilterType);

            LabelsListHeaderState headerState =
                LabelsListHeaderState.GetDefault();

            TreeHeaderSettings.Load(
                headerState,
                UnityConstants.LABELS_TABLE_SETTINGS_NAME,
                (int)LabelsListColumn.Name,
                false);

            mLabelsListView = new LabelsListView(
                headerState,
                LabelsListHeaderState.GetColumnNames(),
                new LabelsViewMenu(this),
                fillLabelsView,
                selectionChangedAction: OnSelectionChanged,
                doubleClickAction: ((ILabelMenuOperations)this).BrowseRepositoryOnLabel,
                afterItemsChangedAction: fillLabelsView.ShowContentOrEmptyState);

            mLabelsListView.Reload();

            mBrowseRepositoryPanel = new BrowseRepositoryPanel(
                wkInfo,
                fillLabelsView,
                parentWindow);
        }

        SearchField mSearchField;
        DateFilter mDateFilter;
        LabelsListView mLabelsListView;

        readonly LabelOperations mLabelOperations;
        readonly LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;
        BrowseRepositoryPanel mBrowseRepositoryPanel;

        readonly object mSplitterState;
        readonly LaunchTool.IProcessExecutor mProcessExecutor;

        readonly EmptyStatePanel mEmptyStatePanel;
        readonly FillLabelsView mFillLabelsView;

        readonly bool mIsGluonMode;
        readonly ViewHost mViewHost;
        readonly IGluonUpdateReport mGluonUpdateReport;
        readonly WorkspaceWindow mWorkspaceWindow;
        readonly ProgressControlsForViews mProgressControls;
        readonly EditorWindow mParentWindow;
        readonly WorkspaceInfo mWkInfo;
        readonly IPendingChangesUpdater mPendingChangesUpdater;
        readonly GluonIncomingChangesUpdater mGluonIncomingChangesUpdater;
        readonly IShelvePendingChangesQuestionerBuilder mShelvePendingChangesQuestionerBuilder;
        readonly IShelvedChangesUpdater mShelvedChangesUpdater;
        readonly IAssetStatusCache mAssetStatusCache;
        readonly SwitchAndShelve.IEnableSwitchAndShelveFeatureDialog mEnableSwitchAndShelveFeatureDialog;
    }
}
