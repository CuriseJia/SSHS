using System;
using System.Reflection;

using UnityEditor;
using UnityEngine;

namespace Unity.PlasticSCM.Editor.UI.UndoRedo
{
    internal class UndoRedoTextArea : UndoRedoHelper.IUndoRedoHost
    {
        internal UndoRedoTextArea(
            Action repaint,
            string text = null,
            string watermark = null)
        {
            mRepaint = repaint;
            mControlName = Guid.NewGuid().ToString();

            mUndoRedoHelper = new UndoRedoHelper(this);

            Text = text;
            WatermarkText = watermark;
        }

        internal string Text
        {
            get { return mText; }
            set
            {
                mRequireFocusReset = true;
                mText = value;
                mCaretIndex = value != null ? value.Length : 0;

                OnTextChanged();
                mRepaint();
            }
        }

        internal string WatermarkText
        {
            get { return mWatermarkText; }
            set
            {
                mWatermarkText = value;
                mRepaint();
            }
        }

        internal int CaretIndex
        {
            get { return mCaretIndex; }
        }

        UndoRedoState UndoRedoHelper.IUndoRedoHost.UndoRedoState
        {
            get { return new UndoRedoState(mText, mCaretIndex); }
            set
            {
                mText = value.Text;
                mCaretIndex = value.CaretPosition;

                TextEditor editor = GetActiveTextEditor();

                if (editor == null)
                    return;

                editor.text = value.Text;
                editor.cursorIndex = value.CaretPosition;
                editor.selectIndex = mCaretIndex;
            }
        }

        internal void OnGUI(GUIStyle watermarkStyle, params GUILayoutOption[] options)
        {
            OnGUIInternal(
                () => EditorGUILayout.TextArea(mText, options),
                () => EditorGUI.LabelField(GUILayoutUtility.GetLastRect(), mWatermarkText, watermarkStyle));
        }

        internal void OnGUI(Rect position, GUIStyle style, GUIStyle watermarkStyle)
        {
            OnGUIInternal(
                () => EditorGUI.TextArea(position, mText, style),
                () => EditorGUI.LabelField(position, mWatermarkText, watermarkStyle));
        }

        void OnGUIInternal(Func<string> drawTextArea, Action drawWatermark)
        {
            GUI.SetNextControlName(mControlName);

            // When the text area has focus, it won't reflect changes to the source string.
            // To update the text programmatically, we must:
            // 1. Remove focus
            // 2. Set the new text
            // 3. Restore focus
            int keyboardControlBackup = GUIUtility.keyboardControl;

            if (mRequireFocusReset && IsFocused(mControlName))
                GUIUtility.keyboardControl = 0;

            string oldText = mText;
            string newText = drawTextArea();
            mText = newText;

            if (string.IsNullOrEmpty(mText) && !string.IsNullOrEmpty(mWatermarkText))
                drawWatermark();

            if (mRequireFocusReset && IsFocused(mControlName))
                GUIUtility.keyboardControl = keyboardControlBackup;

            mRequireFocusReset = false;

            TextEditor editor = GetActiveTextEditor();

            if (editor != null)
            {
                int oldCaretIndex = mCaretIndex;
                int newCaretIndex = editor.cursorIndex;
                mCaretIndex = newCaretIndex;

                if (oldCaretIndex != newCaretIndex && Event.current.type != EventType.Used)
                    OnCaretIndexChanged();
            }

            if (oldText != newText)
                OnTextChanged();

            if (!IsFocused(mControlName))
                return;

            ProcessKeyboardShorcuts(mUndoRedoHelper, Event.current);
        }

        void OnCaretIndexChanged()
        {
            UndoRedoState state;
            if (!mUndoRedoHelper.TryGetLastState(out state))
                return;

            if (state.Text == Text)
                mUndoRedoHelper.UpdateLastState();
        }

        void OnTextChanged()
        {
            mUndoRedoHelper.Snapshot();
        }

        static void ProcessKeyboardShorcuts(UndoRedoHelper undoRedoHelper, Event e)
        {
            if (IsUndoShortcutPressed(e))
            {
                e.Use();

                if (!undoRedoHelper.CanUndo)
                    return;

                undoRedoHelper.Undo();
                return;
            }

            if (IsRedoShortcutPressed(e))
            {
                e.Use();

                if (!undoRedoHelper.CanRedo)
                    return;

                undoRedoHelper.Redo();
                return;
            }
        }

        static bool IsUndoShortcutPressed(Event e)
        {
            return Keyboard.IsControlOrCommandKeyPressed(e) &&
                   Keyboard.IsKeyPressed(e, KeyCode.Z);
        }

        static bool IsRedoShortcutPressed(Event e)
        {
            if (Keyboard.IsControlOrCommandAndShiftKeyPressed(e) &&
                Keyboard.IsKeyPressed(e, KeyCode.Z))
                return true;

            if (Application.platform != RuntimePlatform.WindowsEditor)
                return false;

            return Keyboard.IsControlOrCommandKeyPressed(e) &&
                   Keyboard.IsKeyPressed(e, KeyCode.Y);
        }

        static TextEditor GetActiveTextEditor()
        {
            // Use reflection to access the internal "activeEditor" field of EditorGUI
            Type editorGUIType = typeof(EditorGUI);
            FieldInfo field = editorGUIType.GetField(
                "activeEditor",
                BindingFlags.Static | BindingFlags.NonPublic);

            if (field != null)
                return field.GetValue(null) as TextEditor;

            return null;
        }

        static bool IsFocused(string controlName)
        {
            return GUI.GetNameOfFocusedControl() == controlName;
        }

        bool mRequireFocusReset;
        int mCaretIndex;
        string mText;
        string mWatermarkText;

        readonly string mControlName;
        readonly Action mRepaint;
        readonly UndoRedoHelper mUndoRedoHelper;
    }
}
