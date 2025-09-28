using System;

using UnityEditor;
using UnityEngine;

using Unity.PlasticSCM.Editor.UI;

namespace Unity.Cloud.Collaborate
{
    internal class UVCSToolbarButton : SubToolbar
    {
        internal UVCSToolbarButton(
            Action<Rect> buttonClicked,
            Action repaintToolbar)
        {
            mButtonClicked = buttonClicked;
            mRepaintToolbar = repaintToolbar;

            mTruncatedTextGUIContent = new GUIContent();
            mTooltipButtonGUIContent = new GUIContent();
        }

        internal bool IsVisible
        {
            get { return mIsVisible; }
            set
            {
                if (mIsVisible == value)
                    return;

                mIsVisible = value;
                InvalidateLayout();
            }
        }

        internal int MaxTextLength
        {
            get { return mMaxTextLenght; }
            set
            {
                if (mMaxTextLenght == value || value < 6)
                    return;

                mMaxTextLenght = value;
                mTruncatedTextGUIContent.text = Truncate(mText, MaxTextLength);
                InvalidateLayout();
            }
        }

        internal string Text
        {
            get { return mText; }
            set
            {
                if (mText == value)
                    return;

                mText = value;
                mTruncatedTextGUIContent.text = Truncate(mText, MaxTextLength);
                InvalidateLayout();
            }
        }

        internal string Tooltip
        {
            get { return mTooltipButtonGUIContent.tooltip; }
            set
            {
                if (mTooltipButtonGUIContent.tooltip == value)
                    return;

                mTooltipButtonGUIContent.tooltip = value;
                RequestRepaint();
            }
        }

        internal Texture LeftIcon
        {
            get { return mLeftIcon; }
            set
            {
                if (mLeftIcon == value)
                    return;

                mLeftIcon = value;
                InvalidateLayout();
            }
        }

        internal Texture RightIcon
        {
            get { return mRightIcon; }
            set
            {
                if (mRightIcon == value)
                    return;

                mRightIcon = value;
                InvalidateLayout();
            }
        }

        internal void BeginUpdate()
        {
            mBatchUpdateCount++;
        }

        internal void EndUpdate()
        {
            mBatchUpdateCount--;

            if (mBatchUpdateCount <= 0)
            {
                mBatchUpdateCount = 0;
                InvalidateLayout();
            }
        }

        void InvalidateLayout()
        {
            if (mBatchUpdateCount > 0)
                return;

            mIsLayoutValid = false;
            mRepaintToolbar();
        }

        void RequestRepaint()
        {
            if (mBatchUpdateCount > 0)
                return;

            mRepaintToolbar();
        }

        public override void OnGUI(Rect rect)
        {
            if (!mIsVisible)
            {
                Width = 0;
                return;
            }

            if (mDropDownIcon == null)
                mDropDownIcon = Images.GetDropDownIcon();

            if (mButtonStyle == null)
                mButtonStyle = UnityStyles.EditorToolbar.Button.AppCmdButton;

            if (mTextStyle == null)
                mTextStyle = UnityStyles.EditorToolbar.Button.ButtonText;

            if (!mIsLayoutValid)
            {
                UpdateLayout();
                Width = mLayout.TotalWidth;
            }

            Rect buttonRect = DrawButton(rect);

            DrawLeftIcon(buttonRect);
            DrawText(buttonRect);
            DrawRightIcon(buttonRect);
            DrawDropDownIcon(buttonRect);
        }

        Rect DrawButton(Rect rect)
        {
            Rect buttonRect = new Rect(
                rect.x,
                rect.y,
                mLayout.TotalWidth,
                rect.height);

            if (GUI.Button(buttonRect, mTooltipButtonGUIContent, mButtonStyle))
            {
                mButtonClicked.Invoke(buttonRect);
            }

            return buttonRect;
        }

        void DrawLeftIcon(Rect buttonRect)
        {
            if (LeftIcon == null)
                return;

            Rect leftIconRect = new Rect(
                mLayout.LeftIconX,
                buttonRect.y + (buttonRect.height - Layout.LEFT_ICON_SIZE) / 2,
                Layout.LEFT_ICON_SIZE,
                Layout.LEFT_ICON_SIZE);

            GUI.DrawTexture(leftIconRect, LeftIcon);
        }

        void DrawText(Rect buttonRect)
        {
            Rect textRect = new Rect(
                mLayout.TextX,
                buttonRect.y + (buttonRect.height - mLayout.TextSize.y) / 2,
                mLayout.TextSize.x,
                mLayout.TextSize.y);

            GUI.Label(textRect, mTruncatedTextGUIContent, mTextStyle);
        }

        void DrawRightIcon(Rect buttonRect)
        {
            if (RightIcon == null)
                return;

            Rect rightIconRect = new Rect(
                mLayout.RightIconX,
                buttonRect.y + (buttonRect.height - Layout.RIGHT_ICON_SIZE) / 2,
                Layout.RIGHT_ICON_SIZE,
                Layout.RIGHT_ICON_SIZE);

            GUI.DrawTexture(rightIconRect, RightIcon);
        }

        void DrawDropDownIcon(Rect buttonRect)
        {
            if (mDropDownIcon == null)
                return;

            Rect dropDownIconRect = new Rect(
                mLayout.DropDownIconX,
                buttonRect.y + (buttonRect.height - Layout.DROPDOWN_ICON_SIZE) / 2,
                Layout.DROPDOWN_ICON_SIZE,
                Layout.DROPDOWN_ICON_SIZE);

            GUI.DrawTexture(dropDownIconRect, mDropDownIcon);
        }

        void UpdateLayout()
        {
            mLayout.TextSize = mTextStyle.CalcSize(mTruncatedTextGUIContent);

            int margin = 2;

            mLayout.LeftIconX = mButtonStyle.margin.left + mButtonStyle.padding.left;
            mLayout.TextX = LeftIcon != null ? mLayout.LeftIconX + Layout.LEFT_ICON_SIZE + margin : mLayout.LeftIconX;
            mLayout.RightIconX = mLayout.TextX + mLayout.TextSize.x + margin;
            mLayout.DropDownIconX =
                RightIcon != null ? mLayout.RightIconX + Layout.RIGHT_ICON_SIZE + margin : mLayout.RightIconX;
            mLayout.TotalWidth =
                mLayout.DropDownIconX +
                (mDropDownIcon != null ? Layout.DROPDOWN_ICON_SIZE : 0) +
                mButtonStyle.margin.right +
                mButtonStyle.margin.right;
        }

        static string Truncate(string text, int maxTextLength)
        {
            const string ellipsis = "...";

            if (text.Length <= maxTextLength)
                return text;

            return string.Concat(text.Substring(0, maxTextLength - ellipsis.Length), ellipsis);
        }

        readonly Action<Rect> mButtonClicked;
        readonly Action mRepaintToolbar;

        readonly GUIContent mTruncatedTextGUIContent;
        readonly GUIContent mTooltipButtonGUIContent;

        bool mIsVisible;
        bool mIsLayoutValid;

        Texture mLeftIcon;
        Texture mRightIcon;
        Texture mDropDownIcon;

        int mMaxTextLenght = 35;
        string mText;
        GUIStyle mButtonStyle;
        GUIStyle mTextStyle;

        Layout mLayout;
        int mBatchUpdateCount;

        struct Layout
        {
            internal const int DROPDOWN_ICON_SIZE = 12;
            internal const int LEFT_ICON_SIZE = 16;
            internal const int RIGHT_ICON_SIZE = 16;

            internal float LeftIconX;
            internal Vector2 TextSize;
            internal float TextX;
            internal float RightIconX;
            internal float DropDownIconX;
            internal float TotalWidth;
        }
    }
}
