using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.Client.Common;
using Codice.CM.Common;
using PlasticGui.CloudDrive.Workspaces;
using PlasticGui.WorkspaceWindow.Items;
using Unity.PlasticSCM.Editor.CloudDrive.Workspaces.DirectoryContent;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;

#if UNITY_6000_2_OR_NEWER
using TreeViewItem = UnityEditor.IMGUI.Controls.TreeViewItem<int>;
#endif

namespace Unity.PlasticSCM.Editor.CloudDrive.Workspaces.Tree
{
    internal class CloudWorkspacesTreeView :
        PlasticTreeView,
        FillCloudWorkspacesView.ICloudWorkspacesTree,
        DirectoryContentPanel.ICloudWorkspacesTreeView
    {
        internal CloudWorkspacesTreeView(Action selectionChangedAction) :
            base(showCustomBackground: false)
        {
            mSelectionChangedAction = selectionChangedAction;

            mDelayedSelectionAction = new DelayedActionBySecondsRunner(
                SelectionChanged, UnityConstants.SELECTION_DELAYED_INPUT_ACTION_INTERVAL);

            rowHeight = 16;
        }

        protected override bool CanChangeExpandedState(TreeViewItem item)
        {
            return ((CloudWorkspacesTreeViewItem)item).IsExpandable;
        }

        protected override IList<TreeViewItem> BuildRows(TreeViewItem rootItem)
        {
            RegenerateRows(
                mExpandedTreesByWorkspace.Values.ToList(),
                mTreeViewItemIds,
                this,
                rootItem,
                mRows);

            if (GetSelectedNode() != null)
                SelectionChanged();

            return mRows;
        }

        public override void OnGUI(Rect rect)
        {
            if (Event.current.type == EventType.Repaint && mLastValidWidth != rect.width)
                mLastValidWidth = rect.width;

            base.OnGUI(rect);
        }

        protected override void RowGUI(RowGUIArgs args)
        {
            if (args.item is CloudWorkspacesTreeViewItem)
            {
                CloudWorkspacesTreeViewItem item = (CloudWorkspacesTreeViewItem)args.item;

                args.rowRect.width = mLastValidWidth;

                WorkspaceContentTreeViewItemGUI(
                    args.rowRect,
                    this,
                    item,
                    GetProgressControlsDataForNode(
                        item.ExpandedTreeNode,
                        mProgressControlsByWorkspace));
                return;
            }

            base.RowGUI(args);
        }

        protected override void SelectionChanged(IList<int> selectedIds)
        {
            mDelayedSelectionAction.Run();
        }

        protected override void ExpandedStateChanged()
        {
            IList<int> expandedIds = GetExpanded();

            ReloadExpandedTreeNodes(
                expandedIds.Except(mLastExpandedIds),
                mTreeViewItemIds,
                mExpandedTreesByWorkspace);

            mLastExpandedIds = expandedIds;
        }

        internal bool IsOperationRunning()
        {
            foreach (ProgressControlsForViews progressControls
                in mProgressControlsByWorkspace.Values)
            {
                if (progressControls.IsOperationRunning())
                    return true;
            }

            return false;
        }

        internal ProgressControlsForViews GetWorkspaceProgressControls(WorkspaceInfo wkInfo)
        {
            return mProgressControlsByWorkspace[wkInfo];
        }

        internal WorkspaceInfo GetSelectedWorkspaceInfo()
        {
            ExpandedTreeNode selectedNode = GetSelectedNode();

            if (selectedNode == null)
                return null;

            return selectedNode.WkInfo;
        }

        internal ExpandedTreeNode GetSelectedNode()
        {
            IList<int> selectedIds = GetSelection();

            if (selectedIds.Count != 1)
                return null;

            int selectedId = selectedIds[0];

            ExpandedTreeNode result;
            if (!mTreeViewItemIds.TryGetItemById(selectedId, out result))
                return null;

            return result;
        }

        internal List<ExpandedTreeNode> GetSelectedNodes()
        {
            List<ExpandedTreeNode> result = new List<ExpandedTreeNode>();

            IList<int> selectedIds = GetSelection();

            foreach (int selectedId in selectedIds)
            {
                ExpandedTreeNode selectedNode;
                if (!mTreeViewItemIds.TryGetItemById(selectedId, out selectedNode))
                    continue;

                result.Add(selectedNode);
            }

            return result;
        }

        internal void Update(EditorWindow parentWindow)
        {
            foreach (ProgressControlsForViews progressControls
                in mProgressControlsByWorkspace.Values)
            {
                progressControls.UpdateProgress(parentWindow);
            }
        }

        void FillCloudWorkspacesView.ICloudWorkspacesTree.Fill(List<ExpandedTree> expandedTrees)
        {
            mExpandedTreesByWorkspace = BuildExpandedTrees(expandedTrees);

            BuildProgressControls(expandedTrees, mProgressControlsByWorkspace);

            Reload();
        }

        void DirectoryContentPanel.ICloudWorkspacesTreeView.SelectNode(
            string wkPath, string fullPath)
        {
            SelectNode(wkPath, fullPath);
        }

        void SelectionChanged()
        {
            if (mIsSelectionChangedEventDisabled)
                return;

            IList<int> selectedIds = GetSelection();

            if (selectedIds.Count == 1)
                ReloadExpandedTreeNodeIfChildrenNotLoaded(
                    selectedIds[0], mTreeViewItemIds, mExpandedTreesByWorkspace);

            mSelectionChangedAction();
        }

        void SelectNode(string wkPath, string fullPath)
        {
            if (ExpandAncestors(wkPath, fullPath, mTreeViewItemIds, this))
                Reload();

            int idToSelect;
            if (!mTreeViewItemIds.TryGetItemIdByKey(fullPath, out idToSelect))
                return;

            mIsSelectionChangedEventDisabled = true;

            try
            {
                TableViewOperations.SetSelectionAndScroll(
                    this, new List<int> { idToSelect });
            }
            finally
            {
                mIsSelectionChangedEventDisabled = false;
            }

            SelectionChanged();
        }

        static void WorkspaceContentTreeViewItemGUI(
            Rect rowRect,
            CloudWorkspacesTreeView tree,
            CloudWorkspacesTreeViewItem item,
            ProgressControlsForViews.Data progressControlsData)
        {
            DrawTreeViewItem.ForIndentedItemWithIcon(
                rowRect,
                item.depth,
                Path.GetFileName(item.ExpandedTreeNode.GetFullPath()),
                string.Empty,
                GetItemIcon(tree, item));

            if (progressControlsData == null || !progressControlsData.IsOperationRunning)
                return;

            DoProgress(rowRect, progressControlsData);
        }

        static void DoProgress(
            Rect rowRect,
            ProgressControlsForViews.Data progressControlsData)
        {
            GUILayout.BeginArea(rowRect);

            GUILayout.BeginHorizontal();

            GUILayout.FlexibleSpace();

            LoadingSpinner.OnGUI(
                progressControlsData.ProgressPercent,
                progressControlsData.ProgressMessage);

            GUILayout.Space(2);

            GUILayout.EndHorizontal();

            GUILayout.EndArea();
        }

        static void ReloadExpandedTreeNodes(
            IEnumerable<int> idsToReload,
            TreeViewItemIds<ExpandedTreeNode> treeViewItemIds,
            Dictionary<WorkspaceInfo, ExpandedTree> expandedTreesByWorkspace)
        {
            foreach (int id in idsToReload)
            {
                ExpandedTreeNode node;
                if (!treeViewItemIds.TryGetItemById(id, out node))
                    continue;

                ExpandedTree expandedTree;
                if (!expandedTreesByWorkspace.TryGetValue(node.WkInfo, out expandedTree))
                    continue;

                expandedTree.ReloadNode(node);
            }
        }

        static void ReloadExpandedTreeNodeIfChildrenNotLoaded(
            int idToReload,
            TreeViewItemIds<ExpandedTreeNode> treeViewItemIds,
            Dictionary<WorkspaceInfo, ExpandedTree> expandedTreesByWorkspace)
        {
            ExpandedTreeNode node;
            if (!treeViewItemIds.TryGetItemById(idToReload, out node))
                return;

            if (node.GetChildren() != null)
                return;

            ExpandedTree expandedTree;
            if (!expandedTreesByWorkspace.TryGetValue(node.WkInfo, out expandedTree))
                return;

            expandedTree.ReloadNode(node);
        }

        static bool ExpandAncestors(
            string wkPath,
            string nodePath,
            TreeViewItemIds<ExpandedTreeNode> treeViewItemIds,
            CloudWorkspacesTreeView treeView)
        {
            List<string> ancestors = GetAncestorsPaths(wkPath, nodePath);
            bool result = false;

            foreach (string ancestor in ancestors)
            {
                int ancestorId;
                if (!treeViewItemIds.TryGetItemIdByKey(ancestor, out ancestorId))
                    continue;

                if (treeView.IsExpanded(ancestorId))
                    continue;

                treeView.SetExpanded(ancestorId, true);
                result = true;
            }

            return result;
        }

        static void RegenerateRows(
            List<ExpandedTree> expandedTrees,
            TreeViewItemIds<ExpandedTreeNode> treeViewItemIds,
            CloudWorkspacesTreeView treeView,
            TreeViewItem rootItem,
            List<TreeViewItem> rows)
        {
            ClearRows(rootItem, rows);

            treeViewItemIds.ClearItems();

            foreach (ExpandedTree expandedTree in expandedTrees)
            {
                ExpandedTreeNode rootNode = expandedTree.GetRootNode();

                if (rootNode == null)
                    continue;

                AddNode(
                    rootNode,
                    expandedTree,
                    treeViewItemIds,
                    treeView,
                    rootItem,
                    rows);
            }
        }

        static void AddNode(
            ExpandedTreeNode node,
            ExpandedTree expandedTree,
            TreeViewItemIds<ExpandedTreeNode> treeViewItemIds,
            CloudWorkspacesTreeView treeView,
            TreeViewItem parentItem,
            List<TreeViewItem> rows)
        {
            int nodeId;
            if (!treeViewItemIds.TryGetItemIdByKey(node.GetFullPath(), out nodeId))
                nodeId = treeViewItemIds.AddItemIdByKey(node.GetFullPath());

            treeViewItemIds.AddItemById(nodeId, node);

            CloudWorkspacesTreeViewItem viewItem =
                new CloudWorkspacesTreeViewItem(
                    nodeId,
                    node,
                    Directory.GetDirectories(node.GetFullPath()).Length > 0,
                    parentItem.depth + 1);

            rows.Add(viewItem);
            parentItem.AddChild(viewItem);

            if (!treeView.IsExpanded(nodeId))
                return;

            if (node.GetChildren() == null)
                expandedTree.ReloadNode(node);

            foreach (ExpandedTreeNode childNode in node.GetChildren())
            {
                if (!ExpandedTreeNode.IsDirectory(childNode))
                    continue;

                AddNode(
                    childNode,
                    expandedTree,
                    treeViewItemIds,
                    treeView,
                    viewItem,
                    rows);
            }
        }

        static void ClearRows(
            TreeViewItem rootItem,
            List<TreeViewItem> rows)
        {
            if (rootItem.hasChildren)
                rootItem.children.Clear();

            rows.Clear();
        }

        static Texture GetItemIcon(
            CloudWorkspacesTreeView tree,
            CloudWorkspacesTreeViewItem item)
        {
            if (item.parent.id == 0)
                return Images.GetCloudWorkspaceIcon();

            if (tree.IsExpanded(item.id))
                return Images.GetFolderOpenedIcon();

            return Images.GetFolderIcon();
        }

        static Dictionary<WorkspaceInfo, ExpandedTree> BuildExpandedTrees(
            List<ExpandedTree> expandedTrees)
        {
            Dictionary<WorkspaceInfo, ExpandedTree> result =
                new Dictionary<WorkspaceInfo, ExpandedTree>();

            foreach (ExpandedTree expandedTree in expandedTrees)
                result[expandedTree.WkInfo] = expandedTree;

            return result;
        }

        static void BuildProgressControls(
            List<ExpandedTree> expandedTrees,
            Dictionary<WorkspaceInfo, ProgressControlsForViews> progressControlsByWorkspace)
        {
            List<WorkspaceInfo> workspaceInfos = new List<WorkspaceInfo>();

            foreach (ExpandedTree expandedTree in expandedTrees)
                workspaceInfos.Add(expandedTree.GetRootNode().WkInfo);

            foreach (WorkspaceInfo wkInfo in progressControlsByWorkspace.Keys)
            {
                if (!workspaceInfos.Contains(wkInfo))
                    progressControlsByWorkspace.Remove(wkInfo);
            }

            foreach (WorkspaceInfo wkInfo in workspaceInfos)
            {
                if (progressControlsByWorkspace.ContainsKey(wkInfo))
                    continue;

                progressControlsByWorkspace[wkInfo] = new ProgressControlsForViews();
            }
        }

        static List<string> GetAncestorsPaths(
            string wkPath,
            string nodePath)
        {
            List<string> ancestors = new List<string>();

            while (!PathHelper.IsSamePath(wkPath, nodePath))
            {
                nodePath = Directory.GetParent(nodePath).FullName;
                ancestors.Add(nodePath);
            }

            ancestors.Reverse();
            return ancestors;
        }

        static ProgressControlsForViews.Data GetProgressControlsDataForNode(
            ExpandedTreeNode node,
            Dictionary<WorkspaceInfo, ProgressControlsForViews> progressControlsByWorkspace)
        {
            if (!ExpandedTreeNode.IsRootNode(node))
                return null;

            ProgressControlsForViews progressControls;
            if (!progressControlsByWorkspace.TryGetValue(
                    node.WkInfo,
                    out progressControls))
                return null;

            return progressControls.ProgressData;
        }

        float mLastValidWidth;
        bool mIsSelectionChangedEventDisabled = false;
        IEnumerable<int> mLastExpandedIds = new List<int>();
        Dictionary<WorkspaceInfo, ExpandedTree> mExpandedTreesByWorkspace =
            new Dictionary<WorkspaceInfo, ExpandedTree>();

        readonly DelayedActionBySecondsRunner mDelayedSelectionAction;
        readonly Action mSelectionChangedAction;
        readonly TreeViewItemIds<ExpandedTreeNode> mTreeViewItemIds =
            new TreeViewItemIds<ExpandedTreeNode>();
        readonly Dictionary<WorkspaceInfo, ProgressControlsForViews> mProgressControlsByWorkspace =
            new Dictionary<WorkspaceInfo, ProgressControlsForViews>();
    }
}
