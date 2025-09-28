using System;
using System.Collections;
using System.Collections.Generic;

using UnityEditor;

using Codice.Client.BaseCommands;
using Codice.Client.Commands.CheckIn;
using Codice.Client.Common;
using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.CM.Common.Replication;
using GluonGui.WorkspaceWindow.Views;
using GluonGui;
using GluonGui.WorkspaceWindow.Views.WorkspaceExplorer.Explorer;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.Topbar;
using PlasticGui.WorkspaceWindow.Replication;
using PlasticGui.WorkspaceWindow.Update;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.Configuration;
using Unity.PlasticSCM.Editor.Developer.UpdateReport;
using Unity.PlasticSCM.Editor.StatusBar;
using Unity.PlasticSCM.Editor.Toolbar;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;

using GluonIncomingChangesUpdater = PlasticGui.Gluon.WorkspaceWindow.IncomingChangesUpdater;
using IGluonUpdateReport = PlasticGui.Gluon.IUpdateReport;
using IGluonWorkspaceStatusChangeListener = PlasticGui.Gluon.IWorkspaceStatusChangeListener;
using GluonUpdateReportDialog = Unity.PlasticSCM.Editor.Gluon.UpdateReport.UpdateReportDialog;

namespace Unity.PlasticSCM.Editor
{
    internal class WorkspaceWindow :
        IWorkspaceWindow,
        IRefreshView,
        IUpdateReport,
        IGluonUpdateReport,
        IGluonWorkspaceStatusChangeListener
    {
        internal WorkspaceStatusString.Data WorkspaceStatus { get; private set; }
        internal string ServerDisplayName { get; private set; }

        internal OperationProgressData Progress { get { return mOperationProgressData; } }

        internal IProgressOperationHandler DeveloperProgressOperationHandler
        {
            get { return mDeveloperProgressOperationHandler; }
        }

        internal Gluon.ProgressOperationHandler GluonProgressOperationHandler
        {
            get { return mGluonProgressOperationHandler; }
        }

        internal WorkspaceWindow(
            WorkspaceInfo wkInfo,
            ViewHost viewHost,
            ViewSwitcher switcher,
            WindowStatusBar windowStatusBar,
            IAssetStatusCache assetStatusCache,
            IMergeViewLauncher mergeViewLauncher,
            PendingChangesUpdater pendingChangesUpdater,
            IncomingChangesUpdater developerIncomingChangesUpdater,
            GluonIncomingChangesUpdater gluonIncomingChangesUpdater,
            ShelvedChangesUpdater shelvedChangesUpdater,
            EditorWindow parentWindow)
        {
            mWkInfo = wkInfo;
            mViewHost = viewHost;
            mSwitcher = switcher;
            mWindowStatusBar = windowStatusBar;
            mAssetStatusCache = assetStatusCache;
            mMergeViewLauncher = mergeViewLauncher;
            mPendingChangesUpdater = pendingChangesUpdater;
            mDeveloperIncomingChangesUpdater = developerIncomingChangesUpdater;
            mGluonIncomingChangesUpdater = gluonIncomingChangesUpdater;
            mShelvedChangesUpdater = shelvedChangesUpdater;
            mParentWindow = parentWindow;
            mGuiMessage = new UnityPlasticGuiMessage();

            mDeveloperProgressOperationHandler = new Developer.ProgressOperationHandler(mWkInfo, this);
            mGluonProgressOperationHandler = new Gluon.ProgressOperationHandler(this);
            mOperationProgressData = new OperationProgressData();

            ((IWorkspaceWindow)this).UpdateTitle();
        }

        internal void SetUpdateNotifierForTesting(UpdateNotifier updateNotifier)
        {
            mUpdateNotifierForTesting = updateNotifier;
        }

        internal void RegisterPendingChangesProgressControls(
            ProgressControlsForViews progressControls)
        {
            mProgressControls = progressControls;
        }

        internal bool IsOperationInProgress()
        {
            return mDeveloperProgressOperationHandler.IsOperationInProgress()
                || mGluonProgressOperationHandler.IsOperationInProgress();
        }

        internal void CancelCurrentOperation()
        {
            if (mDeveloperProgressOperationHandler.IsOperationInProgress())
            {
                mDeveloperProgressOperationHandler.CancelCheckinProgress();
                return;
            }

            if (mGluonProgressOperationHandler.IsOperationInProgress())
            {
                mGluonProgressOperationHandler.CancelUpdateProgress();
                return;
            }
        }

        internal void OnParentUpdated(double elapsedSeconds)
        {
            if (IsOperationInProgress() || mRequestedRepaint)
            {
                if (mDeveloperProgressOperationHandler.IsOperationInProgress())
                    mDeveloperProgressOperationHandler.Update(elapsedSeconds);

                mParentWindow.Repaint();

                mRequestedRepaint = false;
            }
        }

        internal void RequestRepaint()
        {
            mRequestedRepaint = true;
        }

        internal void UpdateWorkspace()
        {
            UpdateWorkspaceOperation update = new UpdateWorkspaceOperation(
                mWkInfo,
                this,
                mSwitcher,
                mMergeViewLauncher,
                this,
                mPendingChangesUpdater,
                mDeveloperIncomingChangesUpdater,
                mShelvedChangesUpdater,
                null);

            update.Run(
                UpdateWorkspaceOperation.UpdateType.UpdateToLatest,
                () => RefreshAsset.UnityAssetDatabase(mAssetStatusCache),
                ShowWorkspaceUpdateSuccess);
        }

        internal void UpdateWorkspaceForMode(bool isGluonMode)
        {
            if (isGluonMode)
            {
                PartialUpdateWorkspace();
                return;
            }

            UpdateWorkspace();
        }

        void IWorkspaceWindow.RefreshView(ViewType viewType)
        {
            mSwitcher.RefreshView(viewType);
        }

        void IWorkspaceWindow.RefreshWorkingObjectViews(WorkingObjectInfo workingObjectInfo)
        {
            mSwitcher.RefreshWorkingObjectViews(workingObjectInfo);
        }

        void IWorkspaceWindow.UpdateTitle()
        {
            RefreshWorkspaceStatus();
            UVCSToolbar.Controller.RefreshWorkspaceWorkingInfo();
        }

        bool IWorkspaceWindow.IsOperationInProgress()
        {
            return IsOperationInProgress();
        }

        bool IWorkspaceWindow.CheckOperationInProgress()
        {
            return ((IProgressOperationHandler)mDeveloperProgressOperationHandler).CheckOperationInProgress();
        }

        void IWorkspaceWindow.ShowUpdateProgress(string title, UpdateNotifier notifier)
        {
            mDeveloperProgressOperationHandler.ShowUpdateProgress(title, mUpdateNotifierForTesting ?? notifier);
        }

        void IWorkspaceWindow.EndUpdateProgress()
        {
            mDeveloperProgressOperationHandler.EndUpdateProgress();
        }

        void IWorkspaceWindow.ShowCheckinProgress()
        {
            mDeveloperProgressOperationHandler.ShowCheckinProgress();
        }

        void IWorkspaceWindow.EndCheckinProgress()
        {
            mDeveloperProgressOperationHandler.EndCheckinProgress();
        }

        void IWorkspaceWindow.RefreshCheckinProgress(
            CheckinStatus checkinStatus,
            BuildProgressSpeedAndRemainingTime.ProgressData progressData)
        {
            mDeveloperProgressOperationHandler.
                RefreshCheckinProgress(checkinStatus, progressData);
        }

        bool IWorkspaceWindow.HasCheckinCancelled()
        {
            return mDeveloperProgressOperationHandler.HasCheckinCancelled();
        }

        void IWorkspaceWindow.ShowReplicationProgress(
            IReplicationOperation replicationOperation)
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.RefreshReplicationProgress(
            BranchReplicationData replicationData,
            ReplicationStatus replicationStatus,
            int current,
            int total)
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.EndReplicationProgress(ReplicationStatus replicationStatus)
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.ShowProgress()
        {
            mDeveloperProgressOperationHandler.ShowProgress();
        }

        void IWorkspaceWindow.ShowProgress(IProgressOperation progressOperation)
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.RefreshProgress(ProgressData progressData)
        {
            mDeveloperProgressOperationHandler.RefreshProgress(progressData);
        }

        void IWorkspaceWindow.EndProgress()
        {
            mDeveloperProgressOperationHandler.EndProgress();
        }

        EncryptionConfigurationDialogData IWorkspaceWindow.RequestEncryptionPassword(string server)
        {
            return EncryptionConfigurationDialog.RequestEncryptionPassword(server, mParentWindow);
        }

        void IWorkspaceWindow.RegisterView(
            ViewType type,
            IDisposable disposable,
            IRefreshableView refreshable,
            IWorkingObjectRefreshableView workingObjectRefreshableView)
        {
        }

        void IWorkspaceWindow.UnregisterView(
            ViewType type,
            IDisposable disposable,
            IRefreshableView refreshable,
            IWorkingObjectRefreshableView workingObjectRefreshableView)
        {
        }

        List<IDisposable> IWorkspaceWindow.GetRegisteredViews()
        {
            return new List<IDisposable>();
        }

        void IRefreshView.ForType(ViewType viewType)
        {
            mSwitcher.RefreshView(viewType);
        }

        void IUpdateReport.Show(WorkspaceInfo wkInfo, IList reportLines)
        {
            UpdateReportDialog.ShowReportDialog(
                wkInfo,
                reportLines,
                mParentWindow);
        }

        void IGluonUpdateReport.AppendReport(string updateReport)
        {
        }

        void IGluonWorkspaceStatusChangeListener.OnWorkspaceStatusChanged()
        {
            RefreshWorkspaceStatus();
            UVCSToolbar.Controller.RefreshWorkspaceWorkingInfo();
            RefreshWorkingObject();
        }

        UpdateReportResult IGluonUpdateReport.ShowUpdateReport(
            WorkspaceInfo wkInfo, List<ErrorMessage> errors)
        {
            return GluonUpdateReportDialog.ShowUpdateReport(
                wkInfo, errors, mParentWindow);
        }

        void ShowWorkspaceUpdateSuccess()
        {
            mWindowStatusBar.Notify(
                new GUIContentNotification(
                    PlasticLocalization.Name.WorkspaceUpdateCompleted.GetString()),
                MessageType.None,
                Images.GetStepOkIcon());
        }

        void RefreshWorkspaceStatus()
        {
            WorkspaceStatusString.Data status = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter();
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    status = WorkspaceStatusString.GetSelectorData(mWkInfo);
                },
                /*afterOperationDelegate*/ delegate
                {
                    if (waiter.Exception != null)
                        return;

                    WorkspaceStatus = status;

                    ServerDisplayName = ResolveServer.ToDisplayString(status.Server);

                    RequestRepaint();
                });
        }

        void RefreshWorkingObject()
        {
            // For partial workspaces the calculation of the working object is just
            // supported for branches, not for changesets
            if (mSwitcher.State.SelectedTab != ViewSwitcher.SelectedTab.Branches)
                return;

            WorkingObjectInfo workingObjectInfo = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter();
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    workingObjectInfo = WorkingObjectInfo.Calculate(mWkInfo);
                },
                /*afterOperationDelegate*/ delegate
                {
                    if (waiter.Exception != null)
                        return;

                    mSwitcher.BranchesTab.SetWorkingObjectInfo(workingObjectInfo.BranchInfo);
                });
        }

        void PartialUpdateWorkspace()
        {
            mProgressControls.ShowProgress(PlasticLocalization.GetString(
                PlasticLocalization.Name.UpdatingWorkspace));

            ((IUpdateProgress)mGluonProgressOperationHandler).ShowCancelableProgress();

            OutOfDateUpdater outOfDateUpdater = new OutOfDateUpdater(mWkInfo, null);

            BuildProgressSpeedAndRemainingTime.ProgressData progressData =
                new BuildProgressSpeedAndRemainingTime.ProgressData(DateTime.Now);

            IThreadWaiter waiter = ThreadWaiter.GetWaiter();
            waiter.Execute(
                /*threadOperationDelegate*/ delegate
                {
                    outOfDateUpdater.Execute();
                },
                /*afterOperationDelegate*/ delegate
                {
                    mProgressControls.HideProgress();

                    ((IUpdateProgress)mGluonProgressOperationHandler).EndProgress();

                    if (mPendingChangesUpdater != null)
                        mPendingChangesUpdater.Update(DateTime.Now);

                    if (mGluonIncomingChangesUpdater != null)
                        mGluonIncomingChangesUpdater.Update(DateTime.Now);

                    RefreshAsset.UnityAssetDatabase(mAssetStatusCache);

                    if (waiter.Exception != null)
                    {
                        ExceptionsHandler.DisplayException(waiter.Exception);
                        return;
                    }

                    ShowUpdateReportDialog(
                        mWkInfo,
                        mViewHost,
                        outOfDateUpdater.Progress,
                        mProgressControls,
                        mGuiMessage,
                        mGluonProgressOperationHandler,
                        this,
                        mGluonIncomingChangesUpdater);
                },
                /*timerTickDelegate*/ delegate
                {
                    UpdateProgress progress = outOfDateUpdater.Progress;

                    if (progress == null)
                        return;

                    if (progress.IsCanceled)
                    {
                        mProgressControls.ShowNotification(
                            PlasticLocalization.GetString(PlasticLocalization.Name.Canceling));
                    }

                    ((IUpdateProgress)mGluonProgressOperationHandler).RefreshProgress(
                        progress,
                        UpdateProgressDataCalculator.CalculateProgressForWorkspaceUpdate(
                            mWkInfo.ClientPath, progress, progressData));
                });
        }

        static void ShowUpdateReportDialog(
            WorkspaceInfo wkInfo,
            ViewHost viewHost,
            UpdateProgress progress,
            IProgressControls progressControls,
            GuiMessage.IGuiMessage guiMessage,
            IUpdateProgress updateProgress,
            IGluonUpdateReport updateReport,
            GluonIncomingChangesUpdater gluonIncomingChangesUpdater)
        {
            if (progress.ErrorMessages.Count == 0)
                return;

            UpdateReportResult updateReportResult =
                updateReport.ShowUpdateReport(wkInfo, progress.ErrorMessages);

            if (!updateReportResult.IsUpdateForcedRequested())
                return;

            UpdateForcedOperation updateForced = new UpdateForcedOperation(
                wkInfo,
                viewHost,
                progress,
                progressControls,
                guiMessage,
                updateProgress,
                updateReport,
                gluonIncomingChangesUpdater);

            updateForced.UpdateForced(
                updateReportResult.UpdateForcedPaths,
                updateReportResult.UnaffectedErrors);
        }

        bool mRequestedRepaint;

        UpdateNotifier mUpdateNotifierForTesting;
        IProgressControls mProgressControls;

        readonly OperationProgressData mOperationProgressData;
        readonly Developer.ProgressOperationHandler mDeveloperProgressOperationHandler;
        readonly Gluon.ProgressOperationHandler mGluonProgressOperationHandler;
        readonly GuiMessage.IGuiMessage mGuiMessage;
        readonly EditorWindow mParentWindow;
        readonly PendingChangesUpdater mPendingChangesUpdater;
        readonly IncomingChangesUpdater mDeveloperIncomingChangesUpdater;
        readonly GluonIncomingChangesUpdater mGluonIncomingChangesUpdater;
        readonly ShelvedChangesUpdater mShelvedChangesUpdater;
        readonly IMergeViewLauncher mMergeViewLauncher;
        readonly IAssetStatusCache mAssetStatusCache;
        readonly ViewSwitcher mSwitcher;
        readonly WindowStatusBar mWindowStatusBar;
        readonly ViewHost mViewHost;
        readonly WorkspaceInfo mWkInfo;
    }
}
