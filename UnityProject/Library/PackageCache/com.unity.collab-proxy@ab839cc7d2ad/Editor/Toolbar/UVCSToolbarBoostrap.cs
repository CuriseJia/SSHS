using UnityEditor;

using Unity.PlasticSCM.Editor.Configuration;
using Unity.PlasticSCM.Editor.Toolbar;

namespace Unity.Cloud.Collaborate
{
    [InitializeOnLoad]
    static class UVCSToolbarBoostrap
    {
        static UVCSToolbarBoostrap()
        {
            if (!ToolConfig.EnableNewUVCSToolbarButtonTokenExists())
                return;

            mDropDownButton = new UVCSToolbarButton(
                UVCSToolbar.Controller.PopupClicked,
                Toolbar.RepaintToolbar);

            Toolbar.AddSubToolbar(mDropDownButton);

            UVCSToolbar.Controller.OnToolbarInvalidated += ToolbarInvalidated;
            UVCSToolbar.Controller.OnToolbarButtonInvalidated += ButtonInvalidated;
        }

        static void ToolbarInvalidated()
        {
            Toolbar.RepaintToolbar();
        }

        static void ButtonInvalidated()
        {
            UVCSToolbarButtonData buttonData = UVCSToolbar.Controller.GetButtonData();

            mDropDownButton.BeginUpdate();

            try
            {
                mDropDownButton.Text = buttonData.Text;
                mDropDownButton.Tooltip = buttonData.Tooltip;
                mDropDownButton.LeftIcon = buttonData.LeftIcon;
                mDropDownButton.RightIcon = buttonData.RightIcon;
                mDropDownButton.IsVisible = buttonData.IsVisible;
            }
            finally
            {
                mDropDownButton.EndUpdate();
            }
        }

        static UVCSToolbarButton mDropDownButton;
    }
}
