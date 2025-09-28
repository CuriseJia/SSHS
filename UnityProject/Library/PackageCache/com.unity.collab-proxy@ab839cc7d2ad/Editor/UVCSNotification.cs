using UnityEngine;

using Codice.Client.BaseCommands;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor
{
    internal class UVCSNotificationStatus
    {
        internal IncomingChangesStatus IncomingChanges { get; set; }

        internal WorkspaceStatusResult WorkspaceStatusResult { get; set; }

        internal enum IncomingChangesStatus
        {
            None,
            Changes,
            Conflicts
        }

        internal Texture GetIcon()
        {
            if (IncomingChanges == IncomingChangesStatus.Changes)
                return Images.GetPlasticNotifyIncomingIcon();

            if (IncomingChanges == IncomingChangesStatus.Conflicts)
                return Images.GetPlasticNotifyConflictIcon();

            if (WorkspaceStatusResult != null)
                return Images.GetPlasticNotifyPendingChangesIcon();

            return Images.GetPlasticViewIcon();
        }

        internal string GetPendingChangesInfoTooltipText()
        {
            if (WorkspaceStatusResult == null)
                return null;

            return PlasticLocalization.Name.PendingChangesInfo.GetString(
                WorkspaceStatusResult.Changes.Count);
        }

        internal void Clean()
        {
            IncomingChanges = IncomingChangesStatus.None;
            WorkspaceStatusResult = null;
        }
    }
}
