using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEngine;

using PlasticGui;
using PlasticGui.WorkspaceWindow.Items;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.CloudDrive.Workspaces.DirectoryContent
{
    internal interface IDirectoryContentMenuOperations
    {
        void Open(ExpandedTreeNode node);
    }

    internal class DirectoryContentPanel : IDirectoryContentMenuOperations
    {
        internal Rect Rect { get { return mLastValidRect; } }
        internal ItemsGridView ItemsGridView { get { return mItemsGridView; } }

        internal interface ICloudWorkspacesTreeView
        {
            void SelectNode(string wkPath, string fullPath);
        }

        internal DirectoryContentPanel(
            ICloudWorkspacesTreeView treeView,
            IDragAndDrop dragAndDrop,
            EditorWindow parentWindow)
        {
            mTreeView = treeView;
            mRepaintAction = parentWindow.Repaint;

            BuildComponents(dragAndDrop, parentWindow);
        }

        internal void OnGUI(Rect rect, bool hasFocus)
        {
            if (Event.current.type == EventType.Repaint && rect != mLastValidRect)
            {
                mLastValidRect = rect;
                mRepaintAction();
            }

            mItemsGridView.OnGUI(
                GetGridAreaRect(mLastValidRect),
                hasFocus);

            EditorGUI.DrawRect(
                GetSeparatorRect(mLastValidRect),
                UnityStyles.Colors.BarBorder);

            DrawItemNameBar.Draw(
                GetItemNameBarRect(mLastValidRect),
                mItemsGridView.GetSelectedItem());
        }

        internal void UpdateItemsForDirectory(ExpandedTreeNode node)
        {
            mItemsDirNode = node;

            string itemsDirPath = node.GetFullPath();

            bool isItemsDirPathChanging =
                 itemsDirPath != mItemsGridView.GetItemsDirPath();

            List<string> itemsPathsToSelect = ItemsGridSelection.
                GetItemsPathsToSelect(isItemsDirPathChanging, mItemsGridView);

            mItemsGridView.UpdateItems(itemsDirPath, node.GetChildren());

            ItemsGridSelection.SelectItems(
                isItemsDirPathChanging,
                itemsPathsToSelect,
                mItemsGridView);

            mRepaintAction();
        }

        internal void CleanItems()
        {
            mItemsGridView.CleanItems();

            mRepaintAction();
        }

        void IDirectoryContentMenuOperations.Open(ExpandedTreeNode node)
        {
            if (node == null)
                return;

            if (!ExpandedTreeNode.IsDirectory(node))
            {
                OpenOperation.OpenFile(node.GetFullPath());
                return;
            }

            mTreeView.SelectNode(node.WkInfo.ClientPath, node.GetFullPath());
        }

        void NavigateBackAction()
        {
            ExpandedTreeNode parentNode = (ExpandedTreeNode)
                ((IPlasticTreeNode)mItemsDirNode).GetParent();

            if (parentNode == null)
                return;

            mTreeView.SelectNode(parentNode.WkInfo.ClientPath, parentNode.GetFullPath());
        }

        void OnItemDoubleClickAction()
        {
            if (mItemsGridView.GetSelectedItemsCount() != 1)
                return;

            ((IDirectoryContentMenuOperations)this).Open(mItemsGridView.GetSelectedItem());
        }

        static Rect GetGridAreaRect(Rect rect)
        {
            return new Rect(
                rect.x, rect.y,
                rect.width, rect.height - ITEM_NAME_BAR_HEIGHT - SEPARATOR_HEIGHT);
        }

        static Rect GetSeparatorRect(Rect rect)
        {
            return new Rect(
                rect.x, GetItemNameBarRect(rect).y - SEPARATOR_HEIGHT,
                rect.width, SEPARATOR_HEIGHT);
        }

        static Rect GetItemNameBarRect(Rect rect)
        {
            return new Rect(
                rect.x, rect.yMax - ITEM_NAME_BAR_HEIGHT,
                rect.width, ITEM_NAME_BAR_HEIGHT);
        }

        void BuildComponents(
            IDragAndDrop dragAndDrop,
            EditorWindow parentWindow)
        {
            mItemsGridView = new ItemsGridView(
                dragAndDrop,
                parentWindow,
                OnItemDoubleClickAction,
                NavigateBackAction);
        }

        ItemsGridView mItemsGridView;
        Rect mLastValidRect;

        ExpandedTreeNode mItemsDirNode;

        readonly Action mRepaintAction;
        readonly ICloudWorkspacesTreeView mTreeView;

        const int SEPARATOR_HEIGHT = 1;
        const int ITEM_NAME_BAR_HEIGHT = 20;
    }
}
