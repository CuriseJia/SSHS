using UnityEditor.IMGUI.Controls;

namespace Unity.PlasticSCM.Editor.Views.Labels
{
    class LabelListViewItem : TreeViewItem
    {
        internal object ObjectInfo { get; private set; }

        internal LabelListViewItem(int id, object objectInfo)
            : base(id, 1)
        {
            ObjectInfo = objectInfo;

            displayName = id.ToString();
        }
    }
}
