using System;
using System.Collections.Generic;
using System.IO;

using UnityEditor;
using UnityEngine;

using Codice.Client.Common;
using Codice.CM.Common;
using PlasticGui;
using PlasticGui.CloudDrive;
using PlasticGui.CloudDrive.Workspaces;
using PlasticGui.WorkspaceWindow.Items;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.CloudDrive.Workspaces.DirectoryContent;
using Unity.PlasticSCM.Editor.CloudDrive.Workspaces.DragAndDrop;
using Unity.PlasticSCM.Editor.CloudDrive.Workspaces.Tree;
using Unity.PlasticSCM.Editor.UI.Progress;

namespace Unity.PlasticSCM.Editor.CloudDrive.Workspaces
{
    internal interface IDragAndDrop
    {
        List<string> GetDragSourcePaths();

        void ClearDragTargetPath();

        void SetDragTargetPath(string path);

        void ExecuteDropAction(string[] paths);
    }

    internal class CloudWorkspacesView :
        IRefreshableView,
        CopyPathsOperation.INotify,
        IDragAndDrop
    {
        internal const float MIN_WIDTH =
            TREE_VIEW_MIN_SIZE +
            SPLITTER_WIDTH +
            DIRECTORY_CONTENT_MIN_SIZE;

        internal CloudWorkspacesView(
            IProgressControls progressControls,
            EditorWindow parentWindow)
        {
            mProgressControls = progressControls;
            mParentWindow = parentWindow;

            mFillCloudWorkspacesView = new FillCloudWorkspacesView();

            BuildComponents(parentWindow);

            mSplitterState = PlasticSplitterGUILayout.InitSplitterState(
                new float[] { 0.25f, 0.75f },
                new int[] { TREE_VIEW_MIN_SIZE, DIRECTORY_CONTENT_MIN_SIZE },
                new int[] { 100000, 100000 }
            );

            UnityEditor.DragAndDrop.AddDropHandler(ProjectBrowserDropHandler);

            Refresh();
        }

        internal bool IsOperationRunning()
        {
            return mCloudWorkspacesTreeView.IsOperationRunning();
        }

        internal void AutoRefresh()
        {
            Refresh();
        }

        internal void OnGUI(Rect rect, bool hasFocus)
        {
            Event e = Event.current;

            if (ProcessTabKeyActionIfNeeded(e, mCloudWorkspacesTreeView))
                e.Use();

            if (ProcessF5KeyActionIfNeeded(e, this))
                e.Use();

            UpdateDirectoryContentFocusIfNeeded(
                e, mCloudWorkspacesTreeView, mDirectoryContentPanel.Rect);

            DoContentArea(
                mCloudWorkspacesTreeView,
                mDirectoryContentPanel,
                mSplitterState,
                rect.width,
                rect.height - EditorStyles.toolbar.fixedHeight,
                hasFocus);

            DoToolbar(GetToolbarRect(rect), this);
        }

        internal void OnEnable()
        {
            UnityEditor.DragAndDrop.AddDropHandler(ProjectBrowserDropHandler);
        }

        internal void OnDisable()
        {
            UnityEditor.DragAndDrop.RemoveDropHandler(ProjectBrowserDropHandler);
        }

        internal void Update(EditorWindow parentWindow)
        {
            mCloudWorkspacesTreeView.Update(parentWindow);
        }

        void IRefreshableView.Refresh()
        {
            Refresh();
        }

        void CopyPathsOperation.INotify.RefreshViewSilently()
        {
            mFillCloudWorkspacesView.FillView(mCloudWorkspacesTreeView, progressControls: null);
        }

        void CopyPathsOperation.INotify.ShowErrors(string wkName, List<ErrorMessage> errors)
        {
            ErrorsDialog.ShowDialog(wkName, errors, mParentWindow);
        }

        List<string> IDragAndDrop.GetDragSourcePaths()
        {
            return mDirectoryContentPanel.ItemsGridView.GetSelectedItemsPaths();
        }

        void IDragAndDrop.ClearDragTargetPath()
        {
            mDragTargetPath = null;
        }

        void IDragAndDrop.SetDragTargetPath(string path)
        {
            mDragTargetPath = path;
        }

        void IDragAndDrop.ExecuteDropAction(string[] paths)
        {
            CopyPaths(
                paths,
                mDragTargetPath,
                mCloudWorkspacesTreeView,
                mParentWindow,
                this,
                beforeStartingOperationDelegate: null,
                endOperationDelegate: null,
                shouldRefreshView: true);
        }

        void Refresh()
        {
            mFillCloudWorkspacesView.FillView(mCloudWorkspacesTreeView, mProgressControls);
        }

        void OnTreeViewSelectionChanged()
        {
            List<ExpandedTreeNode> selectedNodes = CloudWorkspacesSelection.
                GetSelectedNodes(mCloudWorkspacesTreeView);

            if (selectedNodes.Count != 1)
            {
                mDirectoryContentPanel.CleanItems();
                return;
            }

            mDirectoryContentPanel.UpdateItemsForDirectory(selectedNodes[0]);
        }

        DragAndDropVisualMode ProjectBrowserDropHandler(
            int dragInstanceId, string dropUponPath, bool perform)
        {
            if (perform == false)
                return DragAndDropVisualMode.None;

            if (UnityEditor.DragAndDrop.objectReferences.Length == 0 ||
                UnityEditor.DragAndDrop.objectReferences[0] != mParentWindow)
                return DragAndDropVisualMode.None;

            CopyPaths(
                UnityEditor.DragAndDrop.paths,
                Path.GetFullPath(dropUponPath),
                mCloudWorkspacesTreeView,
                mParentWindow,
                this,
                BeforeCopyPathsForProjectBrowserDropAction,
                AfterCopyPathsForProjectBrowserDropAction,
                shouldRefreshView: false);

            return DragAndDropVisualMode.Copy;
        }

        void BeforeCopyPathsForProjectBrowserDropAction()
        {
            if (mCopyPathsForProjectBrowserDropCount == 0)
                RefreshAsset.BeforeLongAssetOperation();

            mCopyPathsForProjectBrowserDropCount++;
        }

        void AfterCopyPathsForProjectBrowserDropAction()
        {
            if (mCopyPathsForProjectBrowserDropCount == 1)
                RefreshAsset.AfterLongAssetOperation();

            mCopyPathsForProjectBrowserDropCount--;
        }

        static void CopyPaths(
            string[] paths,
            string targetPath,
            CloudWorkspacesTreeView cloudWorkspacesTreeView,
            EditorWindow parentWindow,
            CopyPathsOperation.INotify notify,
            Action beforeStartingOperationDelegate,
            Action endOperationDelegate,
            bool shouldRefreshView)
        {
            WorkspaceInfo wkInfo = cloudWorkspacesTreeView.GetSelectedWorkspaceInfo();

            ProgressControlsForViews workspaceProgressControls =
                cloudWorkspacesTreeView.GetWorkspaceProgressControls(wkInfo);

            if (workspaceProgressControls.IsOperationRunning())
            {
                GuiMessage.ShowInformation(PlasticLocalization.Name.OperationInProgress.GetString());
                return;
            }

            CopyPathsOperation.CopyRecursivelyTo(
                paths,
                targetPath,
                Path.GetFileName(wkInfo.ClientPath),
                new AskUserForExistingFile(parentWindow),
                notify,
                workspaceProgressControls,
                beforeStartingOperationDelegate,
                endOperationDelegate,
                shouldRefreshView);
        }

        static bool ProcessTabKeyActionIfNeeded(
            Event e,
            CloudWorkspacesTreeView treeView)
        {
            if (!Keyboard.IsTabPressed(e))
                return false;

            if (!treeView.HasFocus())
            {
                treeView.SetFocus();
                return true;
            }

            GUI.FocusControl(null);
            return true;
        }

        static bool ProcessF5KeyActionIfNeeded(
            Event e,
            IRefreshableView view)
        {
            if (!Keyboard.IsKeyPressed(e, KeyCode.F5))
                return false;

            view.Refresh();
            return true;
        }

        static void UpdateDirectoryContentFocusIfNeeded(
            Event e,
            CloudWorkspacesTreeView treeView,
            Rect directoryContentRect)
        {
            if (e.type != EventType.MouseDown)
                return;

            if (!directoryContentRect.Contains(e.mousePosition))
                return;

            if (!treeView.HasFocus())
                return;

            GUI.FocusControl(null);
        }

        static void DoContentArea(
            CloudWorkspacesTreeView cloudWorkspacesTreeView,
            DirectoryContentPanel directoryContentPanel,
            object splitterState,
            float width,
            float height,
            bool hasFocus)
        {
            PlasticSplitterGUILayout.BeginHorizontalSplit(splitterState);
            Rect treeRect = GetTreeRect(width, height);
            Rect directoryContentPanelRect = GetDirectoryContentPanelRect(width, height);
            PlasticSplitterGUILayout.EndHorizontalSplit();

            cloudWorkspacesTreeView.OnGUI(treeRect);

            directoryContentPanel.OnGUI(
                directoryContentPanelRect,
                hasFocus && !cloudWorkspacesTreeView.HasFocus());

            EditorGUI.DrawRect(
                GetSplitterRect(directoryContentPanel.Rect),
                UnityStyles.Colors.BarBorder);
        }

        static void DoToolbar(Rect rect, IRefreshableView view)
        {
            GUILayout.BeginArea(rect);

            EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);

            GUILayout.FlexibleSpace();

            DoToolbarButton(view);

            EditorGUILayout.EndHorizontal();

            GUILayout.EndArea();
        }

        static void DoToolbarButton(IRefreshableView view)
        {
            if (DrawToolbarButton(
                    Images.GetRefreshIcon(),
                    PlasticLocalization.Name.RefreshButton.GetString()))
                view.Refresh();
        }

        static bool DrawToolbarButton(Texture icon, string tooltip)
        {
            return GUILayout.Button(
                new GUIContent(icon, tooltip),
                EditorStyles.toolbarButton,
                GUILayout.Width(26));
        }

        static Rect GetToolbarRect(Rect rect)
        {
            return new Rect(
                rect.x,
                rect.y,
                rect.width,
                EditorStyles.toolbar.fixedHeight);
        }

        static Rect GetSplitterRect(Rect directoryContentRect)
        {
            return new Rect(
                directoryContentRect.x,
                directoryContentRect.y,
                SPLITTER_WIDTH,
                directoryContentRect.height);
        }

        static Rect GetTreeRect(float width, float height)
        {
            Rect result = GUILayoutUtility.GetRect(
                width, height,
                GUILayout.ExpandWidth(true));

            result.y += EditorStyles.toolbar.fixedHeight;

            return result;
        }

        static Rect GetDirectoryContentPanelRect(float width, float height)
        {
            Rect result = GUILayoutUtility.GetRect(
                width, height,
                GUILayout.ExpandWidth(true));

            result.y += EditorStyles.toolbar.fixedHeight;

            return result;
        }

        void BuildComponents(EditorWindow parentWindow)
        {
            mCloudWorkspacesTreeView = new CloudWorkspacesTreeView(
                OnTreeViewSelectionChanged);
            mCloudWorkspacesTreeView.Reload();

            mDirectoryContentPanel = new DirectoryContentPanel(
                mCloudWorkspacesTreeView, this, parentWindow);
        }

        int mCopyPathsForProjectBrowserDropCount = 0;
        string mDragTargetPath;

        object mSplitterState;
        CloudWorkspacesTreeView mCloudWorkspacesTreeView;
        DirectoryContentPanel mDirectoryContentPanel;

        readonly FillCloudWorkspacesView mFillCloudWorkspacesView;
        readonly IProgressControls mProgressControls;
        readonly EditorWindow mParentWindow;

        const float SPLITTER_WIDTH = 1;
        const int TREE_VIEW_MIN_SIZE = 139;
        const int DIRECTORY_CONTENT_MIN_SIZE =
            2 * (int)DrawItemsGridView.GRID_AREA_MARGIN +
            (int)DrawExpandedTreeNode.ICON_SIZE;
    }
}
