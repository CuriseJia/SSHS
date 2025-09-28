using System.Collections.Generic;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.CM.Common;
using Codice.CM.Common.Mount;
using GluonGui.WorkspaceWindow.Views.WorkspaceExplorer.Explorer;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.QueryViews;
using PlasticGui.WorkspaceWindow.QueryViews.Shelves;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;

using GluonIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.IncomingChangesUpdater;
using Unity.PlasticSCM.Editor.Views.Diff;
using GluonShelveOperations = GluonGui.WorkspaceWindow.Views.Shelves.ShelveOperations;

namespace Unity.PlasticSCM.Editor.Views.Shelves
{
    internal partial class ShelvesTab :
        IRefreshableView,
        IShelveMenuOperations,
        IGetQueryText,
        IGetFilter,
        FillShelvesView.IShowContentView
    {
        internal string EmptyStateMessage { get { return mEmptyStatePanel.Text; } }
        internal ShelvesListView Table { get { return mShelvesListView; } }
        internal IShelveMenuOperations Operations { get { return this; } }
        internal IProgressControls ProgressControls { get { return mProgressControls; } }
        internal DiffPanel DiffPanel { get { return mDiffPanel; } }

        internal ShelvesTab(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            ChangesetInfo shelveToSelect,
            WorkspaceWindow workspaceWindow,
            IViewSwitcher viewSwitcher,
            IMergeViewLauncher mergeViewLauncher,
            IHistoryViewLauncher historyViewLauncher,
            GluonShelveOperations.ICheckinView pendingChangesTab,
            IProgressOperationHandler progressOperationHandler,
            IUpdateProgress updateProgress,
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
            bool isGluonMode)
        {
            mWkInfo = wkInfo;
            mRepSpec = repSpec;
            mRefreshView = workspaceWindow;
            mMergeViewLauncher = mergeViewLauncher;
            mPendingChangesTab = pendingChangesTab;
            mProgressOperationHandler = progressOperationHandler;
            mUpdateProgress = updateProgress;
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mAssetStatusCache = assetStatusCache;
            mSaveAssets = saveAssets;
            mShowDownloadPlasticExeWindow = showDownloadPlasticExeWindow;
            mProcessExecutor = processExecutor;
            mWorkspaceOperationsMonitor = workspaceOperationsMonitor;
            mParentWindow = parentWindow;
            mIsGluonMode = isGluonMode;

            mEmptyStatePanel = new EmptyStatePanel(parentWindow.Repaint);

            mProgressControls = new ProgressControlsForViews();

            mFillShelvesView = new FillShelvesView(
                wkInfo,
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
                mFillShelvesView);

            mSplitterState = PlasticSplitterGUILayout.InitSplitterState(
                new float[] { 0.50f, 0.50f },
                new int[] { 100, (int)UnityConstants.DIFF_PANEL_MIN_WIDTH },
                new int[] { 100000, 100000 }
            );

            RefreshAndSelect(shelveToSelect);
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
                mShelvesListView.multiColumnHeader.state,
                UnityConstants.SHELVES_TABLE_SETTINGS_NAME);
        }

        internal void Update()
        {
            mDiffPanel.Update();

            mProgressControls.UpdateProgress(mParentWindow);
        }

        internal void OnGUI()
        {
            DoActionsToolbar(mProgressControls);

            PlasticSplitterGUILayout.BeginHorizontalSplit(mSplitterState);

            DoShelvesArea(
                mShelvesListView,
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
                mShelvesListView,
                UnityConstants.SEARCH_FIELD_WIDTH);
        }

        internal void DrawOwnerFilter()
        {
            GUI.enabled = !mProgressControls.IsOperationRunning();

            EditorGUI.BeginChangeCheck();

            mOwnerFilter = (OwnerFilter)
                EditorGUILayout.EnumPopup(
                    mOwnerFilter,
                    EditorStyles.toolbarDropDown,
                    GUILayout.Width(100));

            if (EditorGUI.EndChangeCheck())
            {
                EnumPopupSetting<OwnerFilter>.Save(
                    mOwnerFilter,
                    UnityConstants.SHELVES_OWNER_FILTER_SETTING_NAME);

                ((IRefreshableView)this).Refresh();
            }

            GUI.enabled = true;
        }

        void IRefreshableView.Refresh()
        {
            RefreshAndSelect(null);
        }

        //IQueryRefreshableView
        public void RefreshAndSelect(RepObjectInfo repObj)
        {
            mDiffPanel.ClearInfo();

            mFillShelvesView.FillView(
                mShelvesListView,
                mProgressControls,
                null,
                null,
                (ChangesetInfo)repObj);
        }

        int IShelveMenuOperations.GetSelectedShelvesCount()
        {
            return ShelvesSelection.GetSelectedShelvesCount(mShelvesListView);
        }

        void IShelveMenuOperations.OpenSelectedShelveInNewWindow()
        {
            LaunchDiffOperations.DiffChangeset(
                mShowDownloadPlasticExeWindow,
                mProcessExecutor,
                ShelvesSelection.GetSelectedRepository(mShelvesListView),
                ShelvesSelection.GetSelectedShelve(mShelvesListView),
                mIsGluonMode);
        }

        void IShelveMenuOperations.ApplyShelveInWorkspace()
        {
            bool isCancelled;
            mSaveAssets.UnderWorkspaceWithConfirmation(
                mWkInfo.ClientPath, mWorkspaceOperationsMonitor,
                out isCancelled);

            if (isCancelled)
                return;

            ChangesetInfo shelveToApply = ShelvesSelection.GetSelectedShelve(mShelvesListView);

            if (mIsGluonMode)
            {
                GluonShelveOperations.ApplyPartialShelveset(
                    mWkInfo,
                    shelveToApply,
                    mRefreshView,
                    PlasticExeLauncher.BuildForResolveConflicts(
                        mWkInfo, true, mShowDownloadPlasticExeWindow),
                    this,
                    mProgressControls,
                    mPendingChangesTab,
                    mUpdateProgress,
                    mProgressOperationHandler,
                    mShelvedChangesUpdater);
                return;
            }

            ShelveOperations.ApplyShelveInWorkspace(
                mRepSpec,
                shelveToApply,
                mMergeViewLauncher,
                mProgressOperationHandler);
        }

        void IShelveMenuOperations.DeleteShelve()
        {
            ShelveOperations.DeleteShelve(
                ShelvesSelection.GetSelectedRepositories(mShelvesListView),
                ShelvesSelection.GetSelectedShelves(mShelvesListView),
                this,
                mProgressControls,
                mShelvedChangesUpdater);
        }

        string IGetQueryText.Get()
        {
            return QueryConstants.BuildShelvesQuery(mOwnerFilter == OwnerFilter.MyShelves);
        }

        Filter IGetFilter.Get()
        {
            return new Filter(mShelvesListView.searchString);
        }

        void IGetFilter.Clear()
        {
            // Not used by the Plugin, needed for the Reset filters button
        }

        void FillShelvesView.IShowContentView.ShowContentPanel()
        {
            mEmptyStatePanel.UpdateContent(string.Empty);
        }

        void FillShelvesView.IShowContentView.ShowEmptyStatePanel(
            string explanationText, bool showResetFilterButton)
        {
            mEmptyStatePanel.UpdateContent(explanationText);
        }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mShelvesListView.SetFocusAndEnsureSelectedItem();
        }

        void OnSelectionChanged()
        {
            List<RepObjectInfo> selectedShelves = ShelvesSelection.
                GetSelectedRepObjectInfos(mShelvesListView);

            if (selectedShelves.Count != 1)
                return;

            mDiffPanel.UpdateInfo(
                MountPointWithPath.BuildWorkspaceRootMountPoint(
                    ShelvesSelection.GetSelectedRepository(mShelvesListView)),
                selectedShelves[0]);
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

        static void DoShelvesArea(
            ShelvesListView shelvesListView,
            EmptyStatePanel emptyStatePanel,
            bool isOperationRunning)
        {
            EditorGUILayout.BeginVertical();

            GUI.enabled = !isOperationRunning;

            Rect rect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

            shelvesListView.OnGUI(rect);

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
            FillShelvesView fillShelvesView)
        {
            mSearchField = new SearchField();
            mSearchField.downOrUpArrowKeyPressed += SearchField_OnDownOrUpArrowKeyPressed;

            mOwnerFilter = EnumPopupSetting<OwnerFilter>.Load(
                UnityConstants.SHELVES_OWNER_FILTER_SETTING_NAME,
                OwnerFilter.MyShelves);

            ShelvesListHeaderState headerState =
                ShelvesListHeaderState.GetDefault();

            TreeHeaderSettings.Load(
                headerState,
                UnityConstants.SHELVES_TABLE_SETTINGS_NAME,
                (int)ShelvesListColumn.Name,
                false);

            mShelvesListView = new ShelvesListView(
                headerState,
                ShelvesListHeaderState.GetColumnNames(),
                new ShelvesViewMenu(this),
                fillShelvesView,
                selectionChangedAction: OnSelectionChanged,
                doubleClickAction: ((IShelveMenuOperations)this).OpenSelectedShelveInNewWindow,
                afterItemsChangedAction: fillShelvesView.ShowContentOrEmptyState);

            mShelvesListView.Reload();

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

        internal enum OwnerFilter
        {
            MyShelves,
            AllShelves
        }

        object mSplitterState;
        OwnerFilter mOwnerFilter;
        SearchField mSearchField;
        ShelvesListView mShelvesListView;
        DiffPanel mDiffPanel;
        readonly FillShelvesView mFillShelvesView;
        readonly ProgressControlsForViews mProgressControls;

        readonly EmptyStatePanel mEmptyStatePanel;
        readonly bool mIsGluonMode;
        readonly EditorWindow mParentWindow;
        readonly WorkspaceOperationsMonitor mWorkspaceOperationsMonitor;
        readonly LaunchTool.IProcessExecutor mProcessExecutor;
        readonly ISaveAssets mSaveAssets;
        readonly LaunchTool.IShowDownloadPlasticExeWindow mShowDownloadPlasticExeWindow;
        readonly IAssetStatusCache mAssetStatusCache;
        readonly IShelvedChangesUpdater mShelvedChangesUpdater;
        readonly IUpdateProgress mUpdateProgress;
        readonly IProgressOperationHandler mProgressOperationHandler;
        readonly GluonShelveOperations.ICheckinView mPendingChangesTab;
        readonly IMergeViewLauncher mMergeViewLauncher;
        readonly IRefreshView mRefreshView;
        readonly RepositorySpec mRepSpec;
        readonly WorkspaceInfo mWkInfo;
    }
}
