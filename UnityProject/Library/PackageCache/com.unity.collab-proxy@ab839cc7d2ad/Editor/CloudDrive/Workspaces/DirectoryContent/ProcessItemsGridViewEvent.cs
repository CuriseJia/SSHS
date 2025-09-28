using System;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;
using UnityEditor;

using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.CloudDrive.Workspaces.DirectoryContent
{
    internal static class ProcessItemsGridViewEvent
    {
        internal const int VERTICAL_SCROLL_STEP = 20;

        internal static bool IfNeeded(
            Event e,
            IItemsGridView itemsGridView,
            IDragAndDrop dragAndDrop,
            EditorWindow parentWindow,
            Action doubleClickAction,
            Action navigateBackAction,
            List<int> selectedItemsIndexes,
            Rect rect,
            Vector2 itemSize,
            float scrollPosition,
            int columnsCount,
            int itemsCount,
            bool isScrollbarVisible,
            bool isDragAvailable)
        {
            if (isScrollbarVisible &&
                ProcessScrollWheelIfNeeded(e, itemsGridView, scrollPosition))
                return true;

            if (ProcessMouseClickIfNeeded(
                    e, itemsGridView, doubleClickAction, selectedItemsIndexes, rect,
                    itemSize, scrollPosition, columnsCount, itemsCount, isScrollbarVisible))
                return true;

            if (ProcessKeyDownIfNeeded(
                    e, itemsGridView, doubleClickAction, navigateBackAction,
                    selectedItemsIndexes, columnsCount, itemsCount))
                return true;

            if (ProcessMouseDragIfNeeded(
                    e, itemsGridView, dragAndDrop, parentWindow, rect, itemSize, scrollPosition,
                    columnsCount, itemsCount, isScrollbarVisible, isDragAvailable))
                return true;

            return false;
        }

        static bool ProcessScrollWheelIfNeeded(
            Event e,
            IItemsGridView itemsGridView,
            float scrollPosition)
        {
            if (e.type != EventType.ScrollWheel)
                return false;

            itemsGridView.UpdateScrollPosition(
                scrollPosition + e.delta.y * VERTICAL_SCROLL_STEP);

            return true;
        }

        static bool ProcessMouseClickIfNeeded(
            Event e,
            IItemsGridView itemsGridView,
            Action doubleClickAction,
            List<int> selectedItemsIndexes,
            Rect rect,
            Vector2 itemSize,
            float scrollPosition,
            int columnsCount,
            int itemsCount,
            bool isScrollbarVisible)
        {
            if (e.type != EventType.MouseDown &&
                e.type != EventType.MouseUp)
                return false;

            Vector2 mousePosition = e.mousePosition - rect.position;

            if (!ItemsPosition.IsMousePositionInsideGrid(
                    mousePosition, rect, isScrollbarVisible))
            {
                return false;
            }

            int itemIndexToSelect = ItemsPosition.GetItemIndexFromMousePosition(
                mousePosition, itemSize, scrollPosition, columnsCount, itemsCount);

            bool shouldUpdateSelectionOnMouseUp =
                selectedItemsIndexes.Contains(itemIndexToSelect) &&
                !Keyboard.HasControlOrCommandModifier(e);

            if (ProcessMouseDownIfNeeded(
                    e,
                    itemsGridView,
                    doubleClickAction,
                    selectedItemsIndexes,
                    itemIndexToSelect,
                    !shouldUpdateSelectionOnMouseUp))
                return true;

            if (ProcessMouseUpIfNeeded(
                    e,
                    itemsGridView,
                    selectedItemsIndexes,
                    itemIndexToSelect,
                    shouldUpdateSelectionOnMouseUp))
                return true;

            return false;
        }

        static bool ProcessMouseDownIfNeeded(
            Event e,
            IItemsGridView itemsGridView,
            Action doubleClickAction,
            List<int> selectedItemsIndexes,
            int itemIndexToSelect,
            bool shouldUpdateSelection)
        {
            if (e.type != EventType.MouseDown)
                return false;

            if (Mouse.IsLeftMouseButtonDoubleClicked(e))
            {
                if (itemIndexToSelect == -1)
                    return false;

                doubleClickAction();
                return true;
            }

            if (!shouldUpdateSelection)
                return true;

            HandleMouseSelection(
                itemsGridView, selectedItemsIndexes, itemIndexToSelect,
                Keyboard.HasControlOrCommandModifier(e),
                Keyboard.HasShiftModifier(e));

            return true;
        }

        static bool ProcessMouseUpIfNeeded(
            Event e,
            IItemsGridView itemsGridView,
            List<int> selectedItemsIndexes,
            int itemIndexToSelect,
            bool shouldUpdateSelection)
        {
            if (e.type != EventType.MouseUp)
                return false;

            if (Mouse.IsLeftMouseButtonDoubleClicked(e))
                return false;

            if (!shouldUpdateSelection)
                return true;

            HandleMouseSelection(
                itemsGridView, selectedItemsIndexes, itemIndexToSelect,
                Keyboard.HasControlOrCommandModifier(e),
                Keyboard.HasShiftModifier(e));

            return true;
        }

        static bool ProcessKeyDownIfNeeded(
            Event e,
            IItemsGridView itemsGridView,
            Action doubleClickAction,
            Action navigateBackAction,
            List<int> selectedItemsIndexes,
            int columnsCount,
            int itemsCount)
        {
            if (e.type != EventType.KeyDown)
                return false;

            bool shouldExtendSelection =
                Keyboard.HasControlOrCommandModifier(e) ||
                Keyboard.HasShiftModifier(e);

            switch (e.keyCode)
            {
                case KeyCode.Return:
                case KeyCode.KeypadEnter:
                    doubleClickAction();
                    break;

                case KeyCode.Backspace:
                    navigateBackAction();
                    break;

                case KeyCode.UpArrow:
                    HandleKeyboardSelection(
                        itemsGridView,
                        selectedItemsIndexes,
                        GetItemIndexToSelectForOffset(selectedItemsIndexes, -columnsCount),
                        itemsCount,
                        shouldExtendSelection);
                    break;

                case KeyCode.DownArrow:
                    HandleKeyboardSelection(
                        itemsGridView,
                        selectedItemsIndexes,
                        GetItemIndexToSelectForOffset(selectedItemsIndexes, columnsCount),
                        itemsCount,
                        shouldExtendSelection);
                    break;

                case KeyCode.LeftArrow:
                    HandleKeyboardSelection(
                        itemsGridView,
                        selectedItemsIndexes,
                        GetItemIndexToSelectForOffset(selectedItemsIndexes, -1),
                        itemsCount,
                        shouldExtendSelection);
                    break;

                case KeyCode.RightArrow:
                    HandleKeyboardSelection(
                        itemsGridView,
                        selectedItemsIndexes,
                        GetItemIndexToSelectForOffset(selectedItemsIndexes, 1),
                        itemsCount,
                        shouldExtendSelection);
                    break;
            }

            return true;
        }

        static bool ProcessMouseDragIfNeeded(
            Event e,
            IItemsGridView itemsGridView,
            IDragAndDrop dragAndDrop,
            EditorWindow parentWindow,
            Rect rect,
            Vector2 itemSize,
            float scrollPosition,
            int columnsCount,
            int itemsCount,
            bool isScrollbarVisible,
            bool isDragAvailable)
        {
            if (!isDragAvailable)
                return false;

            if (e.type == EventType.MouseDrag)
            {
                ProcessDragEvent.ProcessMouseDrag(dragAndDrop, parentWindow);
                return true;
            }

            if (e.type == EventType.DragExited)
            {
                ProcessDragEvent.ProcessDragExited(itemsGridView, dragAndDrop);
                return true;
            }

            if (e.type == EventType.DragUpdated)
            {
                ProcessDragEvent.ProcessDragUpdated(
                    itemsGridView,
                    dragAndDrop,
                    parentWindow,
                    e.mousePosition,
                    rect,
                    itemSize,
                    scrollPosition,
                    columnsCount,
                    itemsCount,
                    isScrollbarVisible);
                return true;
            }

            if (e.type == EventType.DragPerform)
            {
                ProcessDragEvent.ProcessDragPerform(
                    itemsGridView,
                    dragAndDrop,
                    parentWindow,
                    e.mousePosition,
                    rect,
                    itemSize,
                    scrollPosition,
                    columnsCount,
                    itemsCount,
                    isScrollbarVisible);
                return true;
            }

            return false;
        }

        static void HandleMouseSelection(
            IItemsGridView itemsGridView,
            List<int> selectedItemsIndexes,
            int itemIndexToSelect,
            bool shouldToggleSelection,
            bool shouldExtendSelection)
        {
            if (itemIndexToSelect == -1)
            {
                itemsGridView.ClearSelectedItems();
                return;
            }

            List<int> indexesToSelect = GetItemsIndexesToSelect(
                selectedItemsIndexes, itemIndexToSelect,
                shouldToggleSelection, shouldExtendSelection);

            if (indexesToSelect.Count == 0)
            {
                itemsGridView.ClearSelectedItems();
                return;
            }

            itemsGridView.SetSelection(indexesToSelect);
        }

        static void HandleKeyboardSelection(
            IItemsGridView itemsGridView,
            List<int> selectedItemsIndexes,
            int itemIndexToSelect,
            int itemsCount,
            bool shouldExtendSelection)
        {
            if (itemIndexToSelect < 0 || itemIndexToSelect >= itemsCount)
                return;

            List<int> indexesToSelect = GetItemsIndexesToSelect(
                selectedItemsIndexes, itemIndexToSelect,
                false, shouldExtendSelection);

            if (indexesToSelect.Count == 0)
            {
                itemsGridView.ClearSelectedItems();
                return;
            }

            itemsGridView.SetSelection(indexesToSelect);
        }

        static int GetItemIndexToSelectForOffset(
            List<int> selectedItemsIndexes,
            int offset)
        {
            if (selectedItemsIndexes.Count == 0)
                return 0;

            return selectedItemsIndexes.Last() + offset;
        }

        static List<int> GetItemsIndexesToSelect(
            List<int> selectedItemsIndexes,
            int indexToSelect,
            bool shouldToggleSelection,
            bool shouldExtendSelection)
        {
            if (shouldExtendSelection)
            {
                return GetItemsIndexesToSelectByExtending(
                    selectedItemsIndexes, indexToSelect);
            }

            if (shouldToggleSelection)
            {
                return GetItemsIndexesToSelectByToggling(
                    selectedItemsIndexes, indexToSelect);
            }

            return new List<int> { indexToSelect };
        }

        static List<int> GetItemsIndexesToSelectByExtending(
            IList<int> selectedItemsIndexes,
            int target)
        {
            int anchor = selectedItemsIndexes.FirstOrDefault();
            int direction = anchor < target ? 1 : -1;
            int count = Math.Abs(target - anchor);

            List<int> result = new List<int>();

            for (int i = 0; i <= count; i++)
                result.Add(anchor + i * direction);

            return result;
        }

        static List<int> GetItemsIndexesToSelectByToggling(
            IList<int> selectedItemsIndexes,
            int target)
        {
            List<int> result = new List<int>(selectedItemsIndexes);

            if (result.Contains(target))
            {
                result.Remove(target);
                return result;
            }

            result.Add(target);
            return result;
        }
    }
}
