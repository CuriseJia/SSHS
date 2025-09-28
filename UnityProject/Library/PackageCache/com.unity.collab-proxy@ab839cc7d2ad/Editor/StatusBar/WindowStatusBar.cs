using System;

using UnityEditor;
using UnityEngine;

using PlasticGui;
using PlasticGui.WorkspaceWindow;
using Unity.PlasticSCM.Editor.Developer;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.StatusBar
{
    internal class WindowStatusBar
    {
        internal interface IIncomingChangesNotification
        {
            bool HasNotification { get; }
            void OnGUI();
            void Show(
                string infoText,
                string actionText,
                string tooltipText,
                bool hasUpdateAction,
                UVCSNotificationStatus.IncomingChangesStatus status);
            void Hide();
        }

        internal class IncomingChangesNotificationData
        {
            internal string InfoText { get; private set; }
            internal string ActionText { get; private set; }
            internal string TooltipText { get; private set; }
            internal bool HasUpdateAction { get; private set; }
            internal UVCSNotificationStatus.IncomingChangesStatus Status { get; private set; }

            internal void UpdateData(
                string infoText,
                string actionText,
                string tooltipText,
                bool hasUpdateAction,
                UVCSNotificationStatus.IncomingChangesStatus status)
            {
                InfoText = infoText;
                ActionText = actionText;
                TooltipText = tooltipText;
                HasUpdateAction = hasUpdateAction;
                Status = status;
            }

            internal void Clear()
            {
                InfoText = string.Empty;
                ActionText = string.Empty;
                TooltipText = string.Empty;
                HasUpdateAction = false;
                Status = UVCSNotificationStatus.IncomingChangesStatus.None;
            }
        }

        internal interface IShelvedChangesNotification :
            CheckShelvedChanges.IUpdateShelvedChangesNotification
        {
            bool HasNotification { get; }
            void SetWorkspaceWindow(
                WorkspaceWindow workspaceWindow);
            void SetShelvedChangesUpdater(
                IShelvedChangesUpdater shelvedChangesUpdater);
            void OnGUI();
        }

        internal void Notify(INotificationContent content, MessageType type, Texture2D image)
        {
            mNotification = new Notification(
                content,
                type,
                image);
            mDelayedNotificationClearAction.Run();
        }

        internal NotificationBar NotificationBar { get; private set; }

        internal WindowStatusBar()
        {
            mDelayedNotificationClearAction = new DelayedActionBySecondsRunner(
                DelayedClearNotification,
                UnityConstants.NOTIFICATION_CLEAR_INTERVAL);

            NotificationBar = new NotificationBar();
        }

        internal void Initialize(
            WorkspaceWindow workspaceWindow,
            IIncomingChangesNotification incomingChangesNotification,
            IShelvedChangesNotification shelvedChangesNotification)
        {
            mWorkspaceWindow = workspaceWindow;
            mIncomingChangesNotification = incomingChangesNotification;
            mShelvedChangesNotification = shelvedChangesNotification;

            if (incomingChangesNotification is IncomingChangesNotification)
                ((IncomingChangesNotification)incomingChangesNotification).SetWorkspaceWindow(workspaceWindow);

            shelvedChangesNotification.SetWorkspaceWindow(workspaceWindow);
        }

        void DelayedClearNotification()
        {
            mNotification = null;
        }

        internal void OnGUI()
        {
            if (NotificationBar.HasNotification &&
                NotificationBar.IsVisible)
            {
                DrawBar(NotificationBar.OnGUI);
            }

            DrawBar(DoContentArea);

            Rect lastRect = GUILayoutUtility.GetLastRect();

            if (MouseEntered(mIsMouseOver, lastRect))
            {
                mIsMouseOver = true;
                mDelayedNotificationClearAction.Pause();
            }

            if (MouseExited(mIsMouseOver, lastRect))
            {
                mIsMouseOver = false;
                mDelayedNotificationClearAction.Resume();
            }
        }

        void DrawBar(Action doContentArea)
        {
            using (new EditorGUILayout.VerticalScope(
                UnityStyles.StatusBar.Bar,
                GUILayout.Height(UnityConstants.STATUS_BAR_HEIGHT)))
            {
                GUILayout.FlexibleSpace();

                using (new EditorGUILayout.HorizontalScope())
                {
                    doContentArea();
                }

                GUILayout.FlexibleSpace();
            }
        }

        void DoContentArea()
        {
            if (NotificationBar.HasNotification)
            {
                DrawNotificationAvailablePanel(NotificationBar);
            }

            if (mIncomingChangesNotification.HasNotification)
            {
                mIncomingChangesNotification.OnGUI();
            }

            if (mShelvedChangesNotification.HasNotification)
            {
                if (mIncomingChangesNotification.HasNotification)
                    EditorGUILayout.Space(15);

                mShelvedChangesNotification.OnGUI();
            }

            if (mNotification != null)
                DrawNotification(mNotification);

            GUILayout.FlexibleSpace();

            DrawWorkspaceStatus(mWorkspaceWindow);
        }

        internal static void DrawNotification(INotificationContent notification)
        {
            GUILayout.BeginVertical();
            GUILayout.FlexibleSpace();

            notification.OnGUI();

            GUILayout.FlexibleSpace();
            GUILayout.EndVertical();
        }

        internal static bool DrawButton(GUIContent content)
        {
            GUIStyle buttonStyle = new GUIStyle(EditorStyles.miniButton);

            Rect rt = GUILayoutUtility.GetRect(
                content,
                buttonStyle,
                GUILayout.Width(60));

            return GUI.Button(
                rt,
                content,
                buttonStyle);
        }

        static void DrawNotificationAvailablePanel(
            NotificationBar notificationBar)
        {
            GUILayout.BeginVertical();
            GUILayout.FlexibleSpace();

            if (GUILayout.Button(PlasticLocalization.GetString(
                    notificationBar.IsVisible ?
                        PlasticLocalization.Name.HideNotification :
                        PlasticLocalization.Name.ShowNotification)))
            {
                notificationBar.SetVisibility(!notificationBar.IsVisible);
            }

            GUILayout.FlexibleSpace();
            GUILayout.EndVertical();
        }

        static void DrawNotification(Notification notification)
        {
            DrawIcon(notification.Image);
            DrawNotification(notification.Content);
        }

        static void DrawWorkspaceStatus(WorkspaceWindow workspaceWindow)
        {
            DrawIcon(Images.GetBranchIcon());

            if (workspaceWindow.WorkspaceStatus == null)
                return;

            DrawWorkspaceStatusLabel(string.Format(
                "{0}@{1}@{2}",
                workspaceWindow.WorkspaceStatus.ObjectSpec,
                workspaceWindow.WorkspaceStatus.RepositoryName,
                workspaceWindow.ServerDisplayName));
        }

        internal static void DrawIcon(Texture2D icon, int size = UnityConstants.STATUS_BAR_ICON_SIZE)
        {
            GUILayout.BeginVertical();
            GUILayout.FlexibleSpace();

            GUILayout.Label(
                icon,
                UnityStyles.StatusBar.Icon,
                GUILayout.Height(size),
                GUILayout.Width(size));

            GUILayout.FlexibleSpace();
            GUILayout.EndVertical();
        }

        static void DrawWorkspaceStatusLabel(string label)
        {
            GUILayout.BeginVertical();
            GUILayout.FlexibleSpace();

            DrawCopyableLabel.For(label, UnityStyles.StatusBar.Label);

            GUILayout.FlexibleSpace();
            GUILayout.EndVertical();
        }

        static bool MouseEntered(
            bool isMouseOver,
            Rect lastRect)
        {
            bool isInside = lastRect.Contains(Event.current.mousePosition);
            return isInside && !isMouseOver;
        }

        static bool MouseExited(
            bool isMouseOver,
            Rect lastRect)
        {
            bool isInside = lastRect.Contains(Event.current.mousePosition);
            return !isInside && isMouseOver;
        }

        class Notification
        {
            internal INotificationContent Content { get; private set; }
            internal MessageType MessageType { get; private set; }
            internal Texture2D Image { get; private set; }

            internal Notification(INotificationContent content, MessageType messageType, Texture2D image)
            {
                Content = content;
                MessageType = messageType;
                Image = image;
            }
        }

        Notification mNotification;
        WorkspaceWindow mWorkspaceWindow;
        IIncomingChangesNotification mIncomingChangesNotification;
        IShelvedChangesNotification mShelvedChangesNotification;
        bool mIsMouseOver = false;

        readonly DelayedActionBySecondsRunner mDelayedNotificationClearAction;
    }
}
