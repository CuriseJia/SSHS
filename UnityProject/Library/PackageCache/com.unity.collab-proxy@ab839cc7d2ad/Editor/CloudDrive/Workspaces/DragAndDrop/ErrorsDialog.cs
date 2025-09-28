using System.Collections.Generic;

using UnityEditor;
using UnityEngine;

using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Errors;

namespace Unity.PlasticSCM.Editor.CloudDrive.Workspaces.DragAndDrop
{
    internal class ErrorsDialog : PlasticDialog
    {
        protected override Rect DefaultRect
        {
            get
            {
                var baseRect = base.DefaultRect;
                return new Rect(baseRect.x, baseRect.y, 810, 385);
            }
        }

        internal static void ShowDialog(
            string workspaceName,
            List<ErrorMessage> errors,
            EditorWindow parentWindow)
        {
            ErrorsDialog dialog = Create(workspaceName, errors);
            dialog.RunModal(parentWindow);
        }

        protected override void OnModalGUI()
        {
            Title(PlasticLocalization.Name.CopyFileErrorsTitle.GetString());

            Paragraph(PlasticLocalization.Name.CopyFileErrorsMessage.GetString(mWorkspaceName));

            DoErrorsArea(mErrorsPanel);

            DoButtonsArea();
        }

        protected override string GetTitle()
        {
            return PlasticLocalization.Name.CopyFileErrorsTitle.GetString();
        }

        static void DoErrorsArea(ErrorsPanel errorsPanel)
        {
            errorsPanel.OnGUI();
        }

        void DoButtonsArea()
        {
            using (new EditorGUILayout.HorizontalScope())
            {
                GUILayout.FlexibleSpace();
                DoCloseButton();
            }
        }

        void DoCloseButton()
        {
            if (!NormalButton(PlasticLocalization.GetString(
                PlasticLocalization.Name.CloseButton)))
                return;

            CancelButtonAction();
        }

        static ErrorsDialog Create(
            string workspaceName,
            List<ErrorMessage> errors)
        {
            var instance = CreateInstance<ErrorsDialog>();
            instance.mEscapeKeyAction = instance.CloseButtonAction;
            instance.mWorkspaceName = workspaceName;
            instance.BuildComponents();
            instance.SetErrorsList(errors);
            return instance;
        }

        void SetErrorsList(List<ErrorMessage> errors)
        {
            mErrorsPanel.UpdateErrorsList(errors);
        }

        void BuildComponents()
        {
            mErrorsPanel = new ErrorsPanel(
                string.Empty,
                UnityConstants.CloudDrive.ERRORS_DIALOG_SETTINGS_NAME);
        }

        ErrorsPanel mErrorsPanel;
        string mWorkspaceName;
    }
}
