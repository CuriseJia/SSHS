using System;
using System.Collections.Generic;

using Codice.Client.BaseCommands;
using Codice.Client.Commands.CheckIn;
using Codice.CM.Common;
using Codice.CM.Common.Replication;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.Replication;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Toolbar.Headless
{
    internal class HeadlessWorkspaceWindow : IWorkspaceWindow
    {
        internal HeadlessWorkspaceWindow(
            IRefreshableView branchesListPopupPanel,
            Action refreshWorkspaceWorkingInfo,
            Action<BranchInfo> setWorkingBranch)
        {
            mBranchesListPopupPanel = branchesListPopupPanel;
            mRefreshWorkspaceWorkingInfo = refreshWorkspaceWorkingInfo;
            mSetWorkingBranch = setWorkingBranch;
            mUpdateProgress = new HeadlessUpdateProgress();
        }

        void IWorkspaceWindow.UpdateTitle()
        {
            UVCSWindow window = GetWindowIfOpened.UVCS();

            if (window == null)
            {
                mRefreshWorkspaceWorkingInfo();
                return;
            }

            window.IWorkspaceWindow.UpdateTitle();
        }

        bool IWorkspaceWindow.CheckOperationInProgress()
        {
            return mUpdateProgress.IsOperationRunning();
        }

        bool IWorkspaceWindow.IsOperationInProgress()
        {
            return mUpdateProgress.IsOperationRunning();
        }

        void IWorkspaceWindow.ShowUpdateProgress(string title, UpdateNotifier notifier)
        {
            mUpdateProgress.ShowUpdateProgress(title, notifier);
        }

        void IWorkspaceWindow.EndUpdateProgress()
        {
            mUpdateProgress.EndUpdateProgress();
        }

        void IWorkspaceWindow.RefreshWorkingObjectViews(WorkingObjectInfo workingObjectInfo)
        {
            if (workingObjectInfo != null && workingObjectInfo.BranchInfo != null)
                mSetWorkingBranch(workingObjectInfo.BranchInfo);

            UVCSWindow window = GetWindowIfOpened.UVCS();

            if (window == null)
                return;

            window.IWorkspaceWindow.RefreshWorkingObjectViews(workingObjectInfo);
        }

        void IWorkspaceWindow.RefreshView(ViewType viewType)
        {
            if (viewType == ViewType.BranchesListPopup)
            {
                mBranchesListPopupPanel.Refresh();
                return;
            }

            UVCSWindow window = GetWindowIfOpened.UVCS();

            if (window == null)
                return;

            window.IWorkspaceWindow.RefreshView(viewType);
        }

        void IWorkspaceWindow.EndCheckinProgress()
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.EndProgress()
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.EndReplicationProgress(ReplicationStatus replicationStatus)
        {
            throw new NotImplementedException();
        }

        List<IDisposable> IWorkspaceWindow.GetRegisteredViews()
        {
            throw new NotImplementedException();
        }

        bool IWorkspaceWindow.HasCheckinCancelled()
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.RefreshCheckinProgress(
            CheckinStatus checkinStatus,
            BuildProgressSpeedAndRemainingTime.ProgressData progressState)
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.RefreshProgress(ProgressData progressData)
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

        void IWorkspaceWindow.RegisterView(
            ViewType type,
            IDisposable disposable,
            IRefreshableView refreshable,
            IWorkingObjectRefreshableView workingObjectRefreshableView)
        {
            throw new NotImplementedException();
        }

        EncryptionConfigurationDialogData IWorkspaceWindow.RequestEncryptionPassword(string server)
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.ShowCheckinProgress()
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.ShowProgress()
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.ShowProgress(IProgressOperation progressOperation)
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.ShowReplicationProgress(IReplicationOperation replicationOperation)
        {
            throw new NotImplementedException();
        }

        void IWorkspaceWindow.UnregisterView(
            ViewType type,
            IDisposable disposable,
            IRefreshableView refreshable,
            IWorkingObjectRefreshableView workingObjectRefreshableView)
        {
            throw new NotImplementedException();
        }

        readonly IRefreshableView mBranchesListPopupPanel;
        readonly Action mRefreshWorkspaceWorkingInfo;
        readonly Action<BranchInfo> mSetWorkingBranch;
        readonly HeadlessUpdateProgress mUpdateProgress;
    }
}
