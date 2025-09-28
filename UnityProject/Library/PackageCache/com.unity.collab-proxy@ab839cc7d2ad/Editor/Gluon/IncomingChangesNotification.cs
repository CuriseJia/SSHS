using UnityEngine;

using Codice.CM.Common;
using PlasticGui.Gluon;
using PlasticGui.Gluon.WorkspaceWindow;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.StatusBar;

namespace Unity.PlasticSCM.Editor.Gluon
{
    internal class IncomingChangesNotification :
        WindowStatusBar.IIncomingChangesNotification
    {
        internal IncomingChangesNotification(
            WorkspaceInfo wkInfo,
            IGluonViewSwitcher gluonViewSwitcher)
        {
            mWkInfo = wkInfo;
            mGluonViewSwitcher = gluonViewSwitcher;
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
                ShowIncomingChanges.FromNotificationBar(mWkInfo, mGluonViewSwitcher);
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

        readonly IGluonViewSwitcher mGluonViewSwitcher;
        readonly WorkspaceInfo mWkInfo;
    }
}
