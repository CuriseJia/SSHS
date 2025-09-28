using UnityEngine;

using PlasticGui;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Toolbar
{
    internal class UVCSToolbarButtonData
    {
        internal bool IsVisible { get; set; }
        internal string Text { get; set; }
        internal string Tooltip { get; set; }
        internal Texture LeftIcon { get; set; }
        internal Texture RightIcon { get; set; }

        internal static UVCSToolbarButtonData BuildDefault()
        {
            return new UVCSToolbarButtonData
            {
                Text = PlasticLocalization.Name.UnityVersionControl.GetString(),
                Tooltip = PlasticLocalization.Name.UseUnityVersionControlToManageYourProject.GetString(),
                LeftIcon = Images.GetPlasticViewIcon(),
                RightIcon = null,
                IsVisible = UVCSToolbarButtonIsShownPreference.IsEnabled(),
            };
        }
    }
}
