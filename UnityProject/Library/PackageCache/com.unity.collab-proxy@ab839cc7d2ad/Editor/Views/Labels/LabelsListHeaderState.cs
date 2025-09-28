using System;
using System.Collections.Generic;

using UnityEditor.IMGUI.Controls;
using UnityEngine;

using PlasticGui;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Tree;

namespace Unity.PlasticSCM.Editor.Views.Labels
{
    internal enum LabelsListColumn
    {
        Name,
        CreationDate,
        Changeset,
        Comment,
        CreatedBy,
        Branch,
        Repository
    }

    [Serializable]
    internal class LabelsListHeaderState : MultiColumnHeaderState, ISerializationCallbackReceiver
    {
        internal static LabelsListHeaderState GetDefault()
        {
            return new LabelsListHeaderState(BuildColumns());
        }

        internal static List<string> GetColumnNames()
        {
            List<string> result = new List<string>();
            result.Add(PlasticLocalization.GetString(PlasticLocalization.Name.NameColumn));
            result.Add(PlasticLocalization.GetString(PlasticLocalization.Name.CreationDateColumn));
            result.Add(PlasticLocalization.GetString(PlasticLocalization.Name.ChangesetColumn));
            result.Add(PlasticLocalization.GetString(PlasticLocalization.Name.CommentColumn));
            result.Add(PlasticLocalization.GetString(PlasticLocalization.Name.CreatedByColumn));
            result.Add(PlasticLocalization.GetString(PlasticLocalization.Name.BranchColumn));
            result.Add(PlasticLocalization.GetString(PlasticLocalization.Name.RepositoryColumn));
            return result;
        }

        internal static string GetColumnName(LabelsListColumn column)
        {
            switch (column)
            {
                case LabelsListColumn.Name:
                    return PlasticLocalization.GetString(PlasticLocalization.Name.NameColumn);
                case LabelsListColumn.CreationDate:
                    return PlasticLocalization.GetString(PlasticLocalization.Name.CreationDateColumn);
                case LabelsListColumn.Changeset:
                    return PlasticLocalization.GetString(PlasticLocalization.Name.ChangesetColumn);
                case LabelsListColumn.Comment:
                    return PlasticLocalization.GetString(PlasticLocalization.Name.CommentColumn);
                case LabelsListColumn.CreatedBy:
                    return PlasticLocalization.GetString(PlasticLocalization.Name.CreatedByColumn);
                case LabelsListColumn.Branch:
                    return PlasticLocalization.GetString(PlasticLocalization.Name.BranchColumn);
                case LabelsListColumn.Repository:
                    return PlasticLocalization.GetString(PlasticLocalization.Name.RepositoryColumn);
                default:
                    return null;
            }
        }

        void ISerializationCallbackReceiver.OnAfterDeserialize()
        {
            if (mHeaderTitles != null)
                TreeHeaderColumns.SetTitles(columns, mHeaderTitles);

            if (mColumnsAllowedToggleVisibility != null)
                TreeHeaderColumns.SetVisibilities(columns, mColumnsAllowedToggleVisibility);
        }

        void ISerializationCallbackReceiver.OnBeforeSerialize()
        {
        }

        static Column[] BuildColumns()
        {
            return new Column[]
            {
                new Column()
                {
                    width = UnityConstants.LabelsColumns.LABELS_NAME_WIDTH,
                    minWidth = UnityConstants.LabelsColumns.LABELS_NAME_MIN_WIDTH,
                    headerContent = new GUIContent(
                        GetColumnName(LabelsListColumn.Name)),
                    allowToggleVisibility = false,
                    sortingArrowAlignment = TextAlignment.Right
                },
                new Column()
                {
                    width = UnityConstants.LabelsColumns.CREATION_DATE_WIDTH,
                    minWidth = UnityConstants.LabelsColumns.CREATION_DATE_MIN_WIDTH,
                    headerContent = new GUIContent(
                        GetColumnName(LabelsListColumn.CreationDate)),
                    sortingArrowAlignment = TextAlignment.Right
                },
                new Column()
                {
                    width = UnityConstants.LabelsColumns.CHANGESET_NUMBER_WIDTH,
                    minWidth = UnityConstants.LabelsColumns.CHANGESET_NUMBER_MIN_WIDTH,
                    headerContent = new GUIContent(
                        GetColumnName(LabelsListColumn.Changeset)),
                    sortingArrowAlignment = TextAlignment.Right
                },
                new Column()
                {
                    width = UnityConstants.LabelsColumns.COMMENT_WIDTH,
                    minWidth = UnityConstants.LabelsColumns.COMMENT_MIN_WIDTH,
                    headerContent = new GUIContent(
                        GetColumnName(LabelsListColumn.Comment)),
                    sortingArrowAlignment = TextAlignment.Right
                },
                new Column()
                {
                    width = UnityConstants.LabelsColumns.CREATEDBY_WIDTH,
                    minWidth = UnityConstants.LabelsColumns.CREATEDBY_MIN_WIDTH,
                    headerContent = new GUIContent(
                        GetColumnName(LabelsListColumn.CreatedBy)),
                    sortingArrowAlignment = TextAlignment.Right
                },
                new Column()
                {
                    width = UnityConstants.LabelsColumns.BRANCH_WIDTH,
                    minWidth = UnityConstants.LabelsColumns.BRANCH_MIN_WIDTH,
                    headerContent = new GUIContent(
                        GetColumnName(LabelsListColumn.Branch)),
                    sortingArrowAlignment = TextAlignment.Right
                },
                new Column()
                {
                    width = UnityConstants.LabelsColumns.REPOSITORY_WIDTH,
                    minWidth = UnityConstants.LabelsColumns.REPOSITORY_MIN_WIDTH,
                    headerContent = new GUIContent(
                        GetColumnName(LabelsListColumn.Repository)),
                    sortingArrowAlignment = TextAlignment.Right
                }
            };
        }

        LabelsListHeaderState(Column[] columns)
            : base(columns)
        {
            if (mHeaderTitles == null)
                mHeaderTitles = TreeHeaderColumns.GetTitles(columns);

            if (mColumnsAllowedToggleVisibility == null)
                mColumnsAllowedToggleVisibility = TreeHeaderColumns.GetVisibilities(columns);
        }

        [SerializeField]
        string[] mHeaderTitles;

        [SerializeField]
        bool[] mColumnsAllowedToggleVisibility;
    }
}
