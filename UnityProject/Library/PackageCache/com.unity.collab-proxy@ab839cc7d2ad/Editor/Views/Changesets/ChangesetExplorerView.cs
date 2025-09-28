using System;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.CM.Common;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.Views.Labels.Dialogs;
using PlasticGui;
using PlasticGui.WorkspaceWindow.QueryViews;
using PlasticGui.WorkspaceWindow.QueryViews.Changesets;

namespace Unity.PlasticSCM.Editor.Views.Changesets
{
    internal class ChangesetExplorerView :
        IGetQueryText,
        IGetFilterText,
        FillChangesetsView.IShowContentView
    {
        internal ChangesetExplorerView(
            CreateLabelDialog parentWindow,
            WorkspaceInfo workspaceInfo,
            ProgressControlsForDialogs progressControls)
        {
            mParentWindow = parentWindow;
            mWkInfo = workspaceInfo;
            mProgressControls = progressControls;
            mEmptyStatePanel = new EmptyStatePanel(parentWindow.Repaint);

            mFillChangesetsView = new FillChangesetsView(
                mWkInfo,
                null,
                null,
                this,
                this,
                this);

            BuildComponents(mFillChangesetsView);

            Refresh();
        }

        internal void OnGUI()
        {
            DoTitleArea();

            bool isEnabled = !mProgressControls.ProgressData.IsWaitingAsyncResult;

            DoToolbarArea(
                mSearchField,
                mChangesetsListView,
                isEnabled,
                Refresh);

            DoListArea(
                mChangesetsListView,
                mEmptyStatePanel,
                isEnabled,
                mParentWindow.Repaint);

            DoButtonsArea();

            mProgressControls.ForcedUpdateProgress(mParentWindow);
        }

        string IGetQueryText.Get()
        {
            return GetChangesetsQuery.For(mDateFilter);
        }

        string IGetFilterText.Get()
        {
            return mChangesetsListView.searchString;
        }

        void IGetFilterText.Clear()
        {
            // Not used by the Plugin, needed for the Reset filters button
        }

        void FillChangesetsView.IShowContentView.ShowContentPanel()
        {
            mEmptyStatePanel.UpdateContent(string.Empty);
        }

        void FillChangesetsView.IShowContentView.ShowEmptyStatePanel(
            string explanationText, bool showResetFilterButton)
        {
            mEmptyStatePanel.UpdateContent(explanationText);
        }

        void SearchField_OnDownOrUpArrowKeyPressed()
        {
            mChangesetsListView.SetFocusAndEnsureSelectedItem();
        }

        void Refresh()
        {
            mFillChangesetsView.FillView(
                mChangesetsListView,
                mProgressControls,
                null,
                null,
                null,
                null);
        }

        void DoTitleArea()
        {
            GUILayout.BeginVertical();

            GUILayout.Label(PlasticLocalization.GetString(
                PlasticLocalization.Name.AvailableChangesets), UnityStyles.Dialog.Title);

            GUILayout.Label(PlasticLocalization.GetString(
                PlasticLocalization.Name.SelectChangesetBelow), UnityStyles.Paragraph);

            GUILayout.Space(10);

            GUILayout.EndVertical();
        }

        void DoToolbarArea(
            SearchField searchField,
            ChangesetsListView listView,
            bool isEnabled,
            Action refreshAction)
        {
            GUILayout.BeginVertical();

            using (new EditorGUILayout.HorizontalScope())
            {
                GUI.enabled = isEnabled;

                if (GUILayout.Button(PlasticLocalization.GetString(
                        PlasticLocalization.Name.RefreshButton), EditorStyles.miniButton))
                {
                    refreshAction();
                }

                DrawDateFilter();

                GUILayout.FlexibleSpace();

                DrawSearchField.For(searchField, listView, SEARCH_FIELD_WIDTH);

                GUI.enabled = true;
            }

            GUILayout.Space(10);

            GUILayout.EndVertical();
        }

        void DrawDateFilter()
        {
            EditorGUI.BeginChangeCheck();

            mDateFilter.FilterType = (DateFilter.Type)
                EditorGUILayout.EnumPopup(mDateFilter.FilterType, GUILayout.Width(100));

            if (EditorGUI.EndChangeCheck())
                Refresh();
        }

        void DoListArea(
            ChangesetsListView listView,
            EmptyStatePanel emptyStatePanel,
            bool isEnabled,
            Action repaint)
        {
            GUILayout.BeginVertical();

            using (new EditorGUILayout.HorizontalScope())
            {
                GUI.enabled = isEnabled;

                Rect treeRect = GUILayoutUtility.GetRect(0, 100000, 0, 100000);

                listView.OnGUI(treeRect);

                if (!emptyStatePanel.IsEmpty())
                    emptyStatePanel.OnGUI(treeRect);

                GUI.enabled = true;

                GUILayout.Space(5);
            }

            GUILayout.Space(10);

            GUILayout.EndVertical();
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                using (new EditorGUILayout.HorizontalScope(GUILayout.MinWidth(450)))
                {
                    GUILayout.Space(2);
                    DrawProgressForDialogs.For(
                        mProgressControls.ProgressData);
                    GUILayout.Space(2);
                }

                GUILayout.FlexibleSpace();

                DoOkButton();
                DoBackButton();
            }
        }

        void DoOkButton()
        {
            if (!GUILayout.Button(PlasticLocalization.GetString(
                    PlasticLocalization.Name.OkButton), UnityStyles.Dialog.NormalButton,
                GUILayout.MinWidth(80),
                GUILayout.Height(25)))
                return;

            OkButtonAction();
        }

        void DoBackButton()
        {
            if (!GUILayout.Button(PlasticLocalization.GetString(
                    PlasticLocalization.Name.BackButton), UnityStyles.Dialog.NormalButton,
                GUILayout.MinWidth(80),
                GUILayout.Height(25)))
                return;

            BackButtonAction();
        }

        void OkButtonAction()
        {
            ChangesetInfo changesetInfo =
                ChangesetsSelection.GetSelectedChangeset(mChangesetsListView);

            if (changesetInfo == null)
                return;

            mParentWindow.SetChangesetId(changesetInfo.ChangesetId);

            mParentWindow.ToggleChangesetExplorer(false);
        }

        void BackButtonAction()
        {
            mParentWindow.ToggleChangesetExplorer(false);
        }

        void BuildComponents(FillChangesetsView fillChangesetsView)
        {
            mSearchField = new SearchField();
            mSearchField.downOrUpArrowKeyPressed += SearchField_OnDownOrUpArrowKeyPressed;

            DateFilter.Type dateFilterType =
                EnumPopupSetting<DateFilter.Type>.Load(
                    UnityConstants.CHANGESETS_DATE_FILTER_SETTING_NAME,
                    DateFilter.Type.LastMonth);
            mDateFilter = new DateFilter(dateFilterType);

            ChangesetsListHeaderState headerState =
                ChangesetsListHeaderState.GetDefault();
            TreeHeaderSettings.Load(headerState,
                UnityConstants.CHANGESETS_TABLE_SETTINGS_NAME,
                (int)ChangesetsListColumn.Name);

            mChangesetsListView = new ChangesetsListView(
                headerState,
                ChangesetsListHeaderState.GetColumnNames(),
                null,
                fillChangesetsView,
                fillChangesetsView,
                selectionChangedAction: () => {},
                doubleClickAction: OkButtonAction,
                afterItemsChangedAction: fillChangesetsView.ShowContentOrEmptyState);

            mChangesetsListView.Reload();
        }

        DateFilter mDateFilter;
        SearchField mSearchField;
        ChangesetsListView mChangesetsListView;

        readonly CreateLabelDialog mParentWindow;
        readonly EmptyStatePanel mEmptyStatePanel;
        readonly FillChangesetsView mFillChangesetsView;
        readonly WorkspaceInfo mWkInfo;
        readonly ProgressControlsForDialogs mProgressControls;

        const float SEARCH_FIELD_WIDTH = 450;
    }
}
