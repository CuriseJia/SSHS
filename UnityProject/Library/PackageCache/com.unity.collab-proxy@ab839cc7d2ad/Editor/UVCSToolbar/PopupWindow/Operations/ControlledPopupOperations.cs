using System;

using UnityEditor;

using UnityEditor.ShortcutManagement;

using Codice.CM.Common;
using GluonGui.WorkspaceWindow.Views.WorkspaceExplorer.Explorer;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.QueryViews.Branches;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;
using Unity.PlasticSCM.Editor.Toolbar.Headless;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.Views;
using Unity.PlasticSCM.Editor.Views.Branches.Dialogs;

namespace Unity.PlasticSCM.Editor.Toolbar.PopupWindow.Operations
{
    internal class ControlledPopupOperations : IQueryRefreshableView
    {
        internal ControlledPopupOperations(
            WorkspaceInfo wkInfo,
            UVCSPlugin uvcsPlugin,
            bool isGluonMode,
            Action refreshWorkspaceWorkingInfo,
            IRefreshableView branchesListPopupPanel,
            Action<BranchInfo> setWorkingBranch,
            Func<RepositorySpec> fetchRepSpec,
            Func<BranchInfo> fetchWorkingBranch)
        {
            mWkInfo = wkInfo;
            mUVCSPlugin = uvcsPlugin;
            mIsGluonMode = isGluonMode;

            mRefreshWorkspaceWorkingInfo = refreshWorkspaceWorkingInfo;
            mBranchesListPopupPanel = branchesListPopupPanel;
            mFetchRepSpec = fetchRepSpec;
            mFetchWorkingBranch = fetchWorkingBranch;
            mWindow = FindEditorWindow.FirstAvailableWindow();

            mSaveAssets = new SaveAssets();
            mShelvePendingChangesQuestionerBuilder = new ShelvePendingChangesQuestionerBuilder(mWindow);
            mEnableSwitchAndShelveFeatureDialog = new EnableSwitchAndShelveFeature(null, mWindow);
            mProgressControls = new HeadlessProgressControls();
            mViewHost = new HeadlessGluonViewHost(branchesListPopupPanel);

            mBranchOperations = new BranchOperations(
                wkInfo,
                new HeadlessWorkspaceWindow(
                    branchesListPopupPanel,
                    refreshWorkspaceWorkingInfo,
                    setWorkingBranch),
                new HeadlessMergeViewLauncher(uvcsPlugin),
                this,
                ViewType.BranchesListPopup,
                mProgressControls,
                new HeadlessUpdateReport(),
                null,                
                new ContinueWithPendingChangesQuestionerBuilder(new HeadlessViewSwitcher(), mWindow),
                mShelvePendingChangesQuestionerBuilder,
                new ApplyShelveWithConflictsQuestionerBuilder(),
                mUVCSPlugin.PendingChangesUpdater,
                mUVCSPlugin.DeveloperIncomingChangesUpdater,
                GetShelvedChangesUpdater(),
                mEnableSwitchAndShelveFeatureDialog);
        }

        internal void ShowPendingChangesView()
        {
            DoShowPendingChangesView(mUVCSPlugin);
        }

        internal void ShowIncomingChangesView()
        {
            DoShowIncomingChangesView(mUVCSPlugin);
        }

        internal void SwitchToBranch(
            BranchInfo branchInfo,
            RepositorySpec repSpec)
        {
            bool isCancelled;
            mSaveAssets.UnderWorkspaceWithConfirmation(
                mWkInfo.ClientPath,
                mUVCSPlugin.WorkspaceOperationsMonitor,
                out isCancelled);

            if (isCancelled)
                return;

            if (mIsGluonMode)
            {
                SwitchToBranchForGluonMode(branchInfo, repSpec);
                return;
            }

            SwitchToBranchForDeveloperMode(branchInfo, repSpec);
        }

        internal void CreateBranch(string proposedBranchName)
        {
            RepositorySpec repSpec = mFetchRepSpec();
            BranchInfo workingBranch = mFetchWorkingBranch();

            if (repSpec == null || workingBranch == null)
                return;

            if (mIsGluonMode)
            {
                CreateBranchForGluonMode(proposedBranchName, repSpec, workingBranch);
                return;
            }

            CreateBranchForDeveloperMode(proposedBranchName, repSpec, workingBranch);
        }

        void IQueryRefreshableView.RefreshAndSelect(RepObjectInfo repObj)
        {
            //TODO: The repObj is not used
            mBranchesListPopupPanel.Refresh();
        }

        void CreateBranchForGluonMode(
            string proposedBranchName,
            RepositorySpec repSpec,
            BranchInfo workingBranch)
        {
            BranchCreationData branchCreationData = CreateBranchDialog.CreateBranchFromLastParentBranchChangeset(
                mWindow,
                repSpec,
                workingBranch,
                proposedBranchName);

            CreateBranchOperation.CreateBranch(
                mWkInfo,
                branchCreationData,
                mViewHost.ViewHost,
                mUVCSPlugin.PendingChangesUpdater,
                mUVCSPlugin.GluonIncomingChangesUpdater,
                new UnityPlasticGuiMessage(),
                mProgressControls,
                new HeadlessGluonUpdateProgress(),
                new HeadlessGluonUpdateReport(),
                new HeadlessWorkspaceStatusChangeListener(mRefreshWorkspaceWorkingInfo),
                mShelvePendingChangesQuestionerBuilder,
                GetShelvedChangesUpdater(),
                mEnableSwitchAndShelveFeatureDialog,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    mUVCSPlugin.AssetStatusCache,
                    ProjectPackages.ShouldBeResolvedFromPaths(mWkInfo, items)));
        }

        void CreateBranchForDeveloperMode(
            string proposedBranchName,
            RepositorySpec repSpec,
            BranchInfo workingBranch)
        {
            BranchCreationData branchCreationData = CreateBranchDialog.CreateBranchFromLastParentBranchChangeset(
                mWindow,
                repSpec,
                workingBranch,
                proposedBranchName);

            mBranchOperations.CreateBranch(
                branchCreationData,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    mUVCSPlugin.AssetStatusCache,
                    ProjectPackages.ShouldBeResolvedFromUpdateReport(mWkInfo, items)));
        }

        void SwitchToBranchForGluonMode(
            BranchInfo branchInfo,
            RepositorySpec repSpec)
        {
            new SwitchToUIOperation().SwitchToBranch(
                mWkInfo,
                branchInfo,
                mViewHost.ViewHost,
                mUVCSPlugin.PendingChangesUpdater,
                mUVCSPlugin.GluonIncomingChangesUpdater,
                new UnityPlasticGuiMessage(),
                mProgressControls,
                new HeadlessGluonUpdateProgress(),
                new HeadlessGluonUpdateReport(),
                new HeadlessWorkspaceStatusChangeListener(mRefreshWorkspaceWorkingInfo),
                mShelvePendingChangesQuestionerBuilder,
                GetShelvedChangesUpdater(),
                mEnableSwitchAndShelveFeatureDialog,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    mUVCSPlugin.AssetStatusCache,
                    ProjectPackages.ShouldBeResolvedFromPaths(mWkInfo, items)));
        }

        void SwitchToBranchForDeveloperMode(
            BranchInfo branchInfo,
            RepositorySpec repSpec)
        {
            mBranchOperations.SwitchToBranch(
                repSpec,
                branchInfo,
                RefreshAsset.BeforeLongAssetOperation,
                items => RefreshAsset.AfterLongAssetOperation(
                    mUVCSPlugin.AssetStatusCache,
                    ProjectPackages.ShouldBeResolvedFromUpdateReport(mWkInfo, items)));
        }

        /*[Shortcut("UVCS/ShowPendingChangesView",
                  ToolbarOperationsShortcut.PendingChangesShortcutKey,
                  ToolbarOperationsShortcut.PendingChangesShortcutModifiers)]*/
        static void DoShowPendingChangesView(UVCSPlugin uvcsPlugin)
        {
            UVCSWindow window = SwitchUVCSPlugin.OnIfNeeded(uvcsPlugin);

            window.ShowPendingChangesView();
        }

        /*[Shortcut("UVCS/ShowIncomingChangesView",
                  ToolbarOperationsShortcut.IncomingChangesShortcutKey,
                  ToolbarOperationsShortcut.IncomingChangesShortcutModifiers)]*/
        static void DoShowIncomingChangesView(UVCSPlugin uvcsPlugin)
        {
            UVCSWindow window = SwitchUVCSPlugin.OnIfNeeded(uvcsPlugin);

            window.ShowIncomingChangesView();
        }

        static IShelvedChangesUpdater GetShelvedChangesUpdater()
        {
            UVCSWindow window = GetWindowIfOpened.UVCS();

            if (window == null)
                return null;

            return window.ShelvedChangesUpdater;
        }

        readonly WorkspaceInfo mWkInfo;
        readonly UVCSPlugin mUVCSPlugin;
        readonly bool mIsGluonMode;
        readonly Action mRefreshWorkspaceWorkingInfo;
        readonly IRefreshableView mBranchesListPopupPanel;
        readonly ISaveAssets mSaveAssets;
        readonly ShelvePendingChangesQuestionerBuilder mShelvePendingChangesQuestionerBuilder;
        readonly EnableSwitchAndShelveFeature mEnableSwitchAndShelveFeatureDialog;
        readonly IProgressControls mProgressControls;
        readonly BranchOperations mBranchOperations;
        readonly Func<RepositorySpec> mFetchRepSpec;
        readonly Func<BranchInfo> mFetchWorkingBranch;
        readonly HeadlessGluonViewHost mViewHost;
        readonly EditorWindow mWindow;
    }
}
