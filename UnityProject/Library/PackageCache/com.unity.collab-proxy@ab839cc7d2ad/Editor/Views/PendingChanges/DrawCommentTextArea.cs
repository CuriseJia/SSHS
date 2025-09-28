using System.Reflection;

using UnityEditor;
using UnityEngine;

using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.UndoRedo;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges
{
    internal static class DrawCommentTextArea
    {
        internal static void For(
            UndoRedoTextArea textArea,
            PendingChangesTab pendingChangesTab,
            float width,
            bool isOperationRunning)
        {
            using (new GuiEnabled(!isOperationRunning))
            {
                EditorGUILayout.BeginHorizontal();

                Rect textAreaRect = BuildTextAreaRect(
                    textArea.Text,
                    width);

                EditorGUI.BeginChangeCheck();

                textArea.OnGUI(
                    textAreaRect,
                    UnityStyles.PendingChangesTab.CommentTextArea,
                    UnityStyles.PendingChangesTab.CommentPlaceHolder);

                if (EditorGUI.EndChangeCheck())
                    OnTextAreaChanged(pendingChangesTab);

                EditorGUILayout.EndHorizontal();
            }
        }

        static void OnTextAreaChanged(PendingChangesTab pendingChangesTab)
        {
            pendingChangesTab.ClearIsCommentWarningNeeded();
        }

        static Rect BuildTextAreaRect(string text, float width)
        {
            GUIStyle commentTextAreaStyle = UnityStyles.PendingChangesTab.CommentTextArea;
            commentTextAreaStyle.stretchWidth = false;

            Rect result = GUILayoutUtility.GetRect(
                width,
                UnityConstants.PENDING_CHANGES_COMMENT_HEIGHT);

            result.width = width;
            result.height = UnityConstants.PENDING_CHANGES_COMMENT_HEIGHT;
            result.xMin = 50f;

            return result;
        }
    }
}
