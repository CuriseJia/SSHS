using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEngine;

using PlasticGui;

namespace Unity.PlasticSCM.Editor.UI
{
    internal abstract class PlasticDialog : EditorWindow, IPlasticDialogCloser
    {
        internal enum SizeToContent
        {
            Manual = 0,
            Automatic = 1
        }

        protected virtual Rect DefaultRect
        {
            get
            {
                int pixelWidth = Screen.currentResolution.width;
                float x = (pixelWidth - DEFAULT_WIDTH) / 2;
                return new Rect(x, 200, DEFAULT_WIDTH, DEFAULT_HEIGHT);
            }
        }

        protected virtual bool IsResizable { get; set; }

        internal void SetSizeToContent(SizeToContent sizeToContent)
        {
            mSizeToContent = sizeToContent;
        }

        internal void OkButtonAction()
        {
            CompleteModal(ResponseType.Ok);
        }

        internal void CancelButtonAction()
        {
            CompleteModal(ResponseType.Cancel);
        }

        internal void CloseButtonAction()
        {
            CompleteModal(ResponseType.None);
        }

        internal void ApplyButtonAction()
        {
            CompleteModal(ResponseType.Apply);
        }

        internal void AddControlConsumingEnterKey(string controlName)
        {
            mControlsConsumingEnterKey.Add(controlName);
        }

        internal void RunUtility(EditorWindow parentWindow)
        {
            InitializeVars(parentWindow);

            if (!IsResizable)
                MakeNonResizable();

            ShowUtility();
        }

        internal ResponseType RunModal(EditorWindow parentWindow)
        {
            InitializeVars(parentWindow);

            if (!IsResizable)
                MakeNonResizable();

            if (UI.RunModal.IsAvailable())
            {
                UI.RunModal.Dialog(this);
                return mAnswer;
            }

            EditorUtility.DisplayDialog(
                PlasticLocalization.GetString(PlasticLocalization.Name.UnityVersionControl),
                PlasticLocalization.GetString(PlasticLocalization.Name.PluginModalInformation),
                PlasticLocalization.GetString(PlasticLocalization.Name.CloseButton));
            return ResponseType.None;
        }

        protected void OnGUI()
        {
            // If the Dialog has been saved into the Unity editor layout and persisted between restarts, the methods
            // to configure the dialogs will be skipped. Simple fix here is to close it when this state is detected.
            // Fixes a NPE loop when the state mentioned above is occurring.
            if (!mIsConfigured)
            {
                Close();
                EditorGUIUtility.ExitGUI();
                return;
            }

            if (!mFocusedOnce)
            {
                // Somehow the prevents the dialog from jumping when dragged
                // NOTE(rafa): We cannot do every frame because the modal kidnaps focus for all processes (in mac at least)
                Focus();
                mFocusedOnce = true;
            }

            ProcessKeyActions();

            GUI.Box(new Rect(0, 0, position.width, position.height), GUIContent.none, EditorStyles.label);

            float margin = 25;
            float marginTop = 25;
            using (new EditorGUILayout.HorizontalScope(GUILayout.Height(position.height)))
            {
                GUILayout.Space(margin);
                using (new EditorGUILayout.VerticalScope(GUILayout.Height(position.height)))
                {
                    GUILayout.Space(marginTop);
                    OnModalGUI();
                    GUILayout.Space(margin);
                }
                GUILayout.Space(margin);
            }

            if (!IsResizable && mSizeToContent == SizeToContent.Automatic)
            {
                var lastRect = GUILayoutUtility.GetLastRect();
                float desiredHeight = lastRect.yMax;

                if (position.height < desiredHeight)
                {
                    Rect newPos = position;
                    newPos.height = desiredHeight;

                    position = newPos;
                }

                maxSize = position.size;
                minSize = maxSize;
            }
        }

        void OnDestroy()
        {
            if (!mIsConfigured)
                return;

            SaveSettings();

            if (mParentWindow == null)
                return;

            mParentWindow.Focus();
        }

        protected virtual void SaveSettings() { }

        protected abstract void OnModalGUI();

        protected abstract string GetTitle();

        protected void Paragraph(string text)
        {
            GUILayout.Label(text, UnityStyles.Paragraph);
            GUILayout.Space(DEFAULT_PARAGRAPH_SPACING);
        }

        protected void TextBlockWithEndLink(
            string url, string formattedExplanation, GUIStyle textblockStyle)
        {
            ExternalLink externalLink = new ExternalLink
            {
                Label = url,
                Url = url
            };

            DrawTextBlockWithLink.ForExternalLink(
                externalLink, formattedExplanation, textblockStyle);
        }

        protected static void Title(string text)
        {
            GUILayout.Label(text, UnityStyles.Dialog.Title);
        }

        protected static bool TitleToggle(string text, bool isOn)
        {
            return EditorGUILayout.ToggleLeft(text, isOn, UnityStyles.Dialog.Title);
        }

        protected static bool TitleToggle(string text, bool isOn, GUIStyle style)
        {
            return EditorGUILayout.ToggleLeft(text, isOn, style);
        }

        protected static bool NormalButton(string text)
        {
            return GUILayout.Button(
                text, UnityStyles.Dialog.NormalButton,
                GUILayout.MinWidth(80),
                GUILayout.Height(25));
        }

        protected static bool AcceptButton(string text, int extraWidth = 10)
        {
            GUI.color = new Color(0.098f, 0.502f, 0.965f, .8f);

            int textWidth = (int)((GUIStyle)UnityStyles.Dialog.AcceptButtonText)
                .CalcSize(new GUIContent(text)).x;

            bool pressed = GUILayout.Button(
                string.Empty, GetEditorSkin().button,
                GUILayout.MinWidth(Math.Max(80, textWidth + extraWidth)),
                GUILayout.Height(25));

            GUI.color = Color.white;

            Rect buttonRect = GUILayoutUtility.GetLastRect();
            GUI.Label(buttonRect, text, UnityStyles.Dialog.AcceptButtonText);

            return pressed;
        }

        void IPlasticDialogCloser.CloseDialog()
        {
            OkButtonAction();
        }

        void ProcessKeyActions()
        {
            Event e = Event.current;

            string focusedControlName = GUI.GetNameOfFocusedControl();

            if (mEnterKeyAction != null &&
                Keyboard.IsReturnOrEnterKeyPressed(e) &&
                !ControlConsumesKey(mControlsConsumingEnterKey, focusedControlName))
            {
                mEnterKeyAction();
                e.Use();
                return;
            }

            if (mEscapeKeyAction != null &&
                Keyboard.IsKeyPressed(e, KeyCode.Escape))
            {
                mEscapeKeyAction();
                e.Use();
                return;
            }
        }

        void CompleteModal(ResponseType answer)
        {
            mAnswer = answer;

            if (mParentWindow == null)
                return;

            Close();
            Repaint();
        }

        void InitializeVars(EditorWindow parentWindow)
        {
            mIsConfigured = true;
            mAnswer = ResponseType.None;

            titleContent = new GUIContent(GetTitle());

            mFocusedOnce = false;

            position = DefaultRect;
            mParentWindow = parentWindow;
        }

        void MakeNonResizable()
        {
            maxSize = DefaultRect.size;
            minSize = maxSize;
        }

        static bool ControlConsumesKey(
            List<string> controlsConsumingKey,
            string focusedControlName)
        {
            if (string.IsNullOrEmpty(focusedControlName))
                return false;

            foreach (string controlName in controlsConsumingKey)
            {
                if (focusedControlName.Equals(controlName))
                    return true;
            }

            return false;
        }

        static GUISkin GetEditorSkin()
        {
            return EditorGUIUtility.isProSkin ?
                EditorGUIUtility.GetBuiltinSkin(EditorSkin.Scene) :
                EditorGUIUtility.GetBuiltinSkin(EditorSkin.Inspector);
        }

        bool mFocusedOnce;
        bool mIsConfigured;
        ResponseType mAnswer;

        protected Action mEnterKeyAction = null;
        protected Action mEscapeKeyAction = null;

        EditorWindow mParentWindow;
        SizeToContent mSizeToContent = SizeToContent.Automatic;

        List<string> mControlsConsumingEnterKey = new List<string>();

        const float DEFAULT_WIDTH = 500f;
        const float DEFAULT_HEIGHT = 180f;
        const float DEFAULT_PARAGRAPH_SPACING = 10f;
    }
}
