using UnityEngine;

using Codice.CM.Common;
using PlasticGui;
using PlasticGui.WorkspaceWindow.PendingChanges;
using Unity.PlasticSCM.Editor.StatusBar;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Developer
{
    internal class IncomingChangesNotification :
        WindowStatusBar.IIncomingChangesNotification
    {
        internal IncomingChangesNotification(
            WorkspaceInfo wkInfo,
            IMergeViewLauncher mergeViewLauncher)
        {
            mWkInfo = wkInfo;
            mMergeViewLauncher = mergeViewLauncher;
        }

        internal void SetWorkspaceWindow(WorkspaceWindow workspaceWindow)
        {
            mWorkspaceWindow = workspaceWindow;
        }

        bool WindowStatusBar.IIncomingChangesNotification.HasNotification
        {
            get { return mHasNotification; }
        }

        void WindowStatusBar.IIncomingChangesNotification.OnGUI()
        {
            Texture2D icon = mData.Status == UVCSNotificationStatus.IncomingChangesStatus.Conflicts ?
                Images.GetConflictedIcon() :
                Images.GetOutOfSyncIcon();

            WindowStatusBar.DrawIcon(icon);

            WindowStatusBar.DrawNotification(new GUIContentNotification(
                new GUIContent(mData.InfoText)));

            if (WindowStatusBar.DrawButton(new GUIContent(mData.ActionText, mData.TooltipText)))
            {
                if (mData.HasUpdateAction)
                {
                    mWorkspaceWindow.UpdateWorkspace();
                    return;
                }

                ShowIncomingChanges.FromNotificationBar(mWkInfo, mMergeViewLauncher);
            }
        }

        void WindowStatusBar.IIncomingChangesNotification.Show(
            string infoText,
            string actionText,
            string tooltipText,
            bool hasUpdateAction,
            UVCSNotificationStatus.IncomingChangesStatus status)
        {
            mData.UpdateData(
                infoText,
                actionText,
                tooltipText,
                hasUpdateAction,
                status);

            mHasNotification = true;
        }

        void WindowStatusBar.IIncomingChangesNotification.Hide()
        {
            mData.Clear();

            mHasNotification = false;
        }

        bool mHasNotification;
        WindowStatusBar.IncomingChangesNotificationData mData =
            new WindowStatusBar.IncomingChangesNotificationData();
        WorkspaceWindow mWorkspaceWindow;

        readonly IMergeViewLauncher mMergeViewLauncher;
        readonly WorkspaceInfo mWkInfo;
    }
}
