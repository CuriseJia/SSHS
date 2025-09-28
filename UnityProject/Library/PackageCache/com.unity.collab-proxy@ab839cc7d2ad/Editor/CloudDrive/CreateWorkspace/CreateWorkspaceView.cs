using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEditor.IMGUI.Controls;
using UnityEngine;

using Codice.Client.Common;
using Codice.Client.Common.WebApi;
using Codice.CM.Common;
using PlasticGui;
using PlasticGui.CloudDrive.Workspaces;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.UI.Tree;

namespace Unity.PlasticSCM.Editor.CloudDrive.CreateWorkspace
{
    internal class CreateWorkspaceView :
        FillCreateWorkspacePanel.INotify,
        IPlasticDialogCloser
    {
        internal interface ICreateWorkspaceListener
        {
            void OnWorkspaceCreated();
        }

        internal CreateWorkspaceView(
            IPlasticWebRestApi restApi,
            IPlasticAPI plasticApi,
            ICreateWorkspaceListener listener,
            EditorWindow parentWindow)
        {
            mPlasticApi = plasticApi;
            mCreateWorkspaceListener = listener;
            mParentWindow = parentWindow;

            mProgressControls = new ProgressControlsForViews();

            BuildComponents(mProgressControls);

            FillCreateWorkspacePanel.LoadOrganizations(
                restApi, plasticApi, mProgressControls, this);
        }

        internal void Update()
        {
            mProgressControls.UpdateProgress(mParentWindow);
        }

        internal void OnGUI()
        {
            DoTitleArea(mExplanation);

            DoEntriesArea(mProgressControls.IsOperationRunning());

            GUILayout.Space(5);

            DoCollaboratorsArea(mProgressControls.IsOperationRunning());

            GUILayout.Space(15);

            DoCreateWorkspaceArea(
                IsInputValid(mSelectedOrganization, mSelectedProject, mWorkspaceName),
                mProgressControls,
                CreateWorkspaceButtonValidationAction);

            GUILayout.Space(15);

            DoNotificationArea(mProgressControls.ProgressData);
        }

        void FillCreateWorkspacePanel.INotify.
            OrganizationsRetrieved(List<string> organizations)
        {
            mOrganizations = organizations;

            string organizationToSelect = GetDefaultValue(mProposedOrganization, mOrganizations);

            if (organizationToSelect == null)
                return;

            OnOrganizationSelected(organizationToSelect);
        }

        void FillCreateWorkspacePanel.INotify.
            ProjectsAndWorkspaceNameRetrieved(List<string> projects, string workspaceName)
        {
            mProjects = projects;
            mWorkspaceName = workspaceName;

            mSelectedProject = GetDefaultValue(mProposedProject, mProjects);

            if (mSelectedProject == null)
                return;

            OnProjectSelected(mSelectedProject);
        }

        void FillCreateWorkspacePanel.INotify.
            WorkspaceNameRetrieved(string workspaceName)
        {
            mWorkspaceName = workspaceName;

            mCollaboratorsListView.Refresh(
                mSelectedOrganization,
                GetProjectGuid(mSelectedOrganization, mSelectedProject));
        }

        void IPlasticDialogCloser.CloseDialog()
        {
            mWkCreationData.Result = true;

            CreateWorkspaceOperation.CreateWorkspace(
                mWkCreationData, mPlasticApi, mProgressControls,
                mCreateWorkspaceListener.OnWorkspaceCreated);
        }

        void OnOrganizationSelected(object organization)
        {
            mSelectedOrganization = organization.ToString();
            mSelectedProject = null;

            mProjects.Clear();

            mExplanation = GetExplanationText(mSelectedOrganization);

            FillCreateWorkspacePanel.LoadProjectsIfNeededAndWorkspaceName(
                mSelectedOrganization, mWorkspaceName ?? mProposedWorkspaceName,
                mPlasticApi, mProgressControls, this);

            mParentWindow.Repaint();
        }

        void OnProjectSelected(object project)
        {
            mSelectedProject = project.ToString();

            mCollaboratorsListView.Refresh(
                mSelectedOrganization,
                GetProjectGuid(mSelectedOrganization, mSelectedProject));

            mParentWindow.Repaint();
        }

        void DoEntriesArea(bool isOperationRunning)
        {
            EditorGUILayout.BeginVertical(GUILayout.MinHeight(108));

            GUI.enabled = !isOperationRunning;

            EntryBuilder.CreateComboBoxEntry(
                PlasticLocalization.Name.OrganizationLabel.GetString(),
                mSelectedOrganization,
                mOrganizations,
                OnOrganizationSelected,
                ENTRY_WIDTH,
                ENTRY_X);

            GUILayout.Space(5);

            if (mSelectedOrganization == null ||
                CloudServer.IsUnityOrganization(mSelectedOrganization))
            {
                EntryBuilder.CreateComboBoxEntry(
                    PlasticLocalization.Name.OrganizationProjectLabel.GetString(),
                    mSelectedProject,
                    mProjects,
                    OnProjectSelected,
                    ENTRY_WIDTH,
                    ENTRY_X);

                DoCreateOrganizationProjectLink(
                    mSelectedOrganization, isOperationRunning);
            }

            GUILayout.Space(5);

            mWorkspaceName = EntryBuilder.CreateTextEntry(
                PlasticLocalization.Name.CloudWorkspaceNameEntry.GetString(),
                mWorkspaceName,
                WKNAME_CONTROL_NAME,
                ENTRY_WIDTH,
                ENTRY_X);

            InputValidationResult inputValidationResult =
                ValidateWorkspaceNameInput(mWorkspaceName);

            if (!inputValidationResult.IsValid)
                DoWarningLabel(inputValidationResult.ErrorMessage, ENTRY_X);

            FocusWorkspaceNameEntryIfNeeded();

            GUI.enabled = true;

            EditorGUILayout.EndVertical();
        }

        void DoCollaboratorsArea(bool isOperationRunning)
        {
            GUILayout.Label(
                PlasticLocalization.Name.ShareDriveTitle.GetString(), UnityStyles.Dialog.Title);

            GUILayout.Space(5);

            GUILayout.Label(
                PlasticLocalization.Name.ShareDriveExplanation.GetString(), UnityStyles.Paragraph);

            GUILayout.Space(10);

            DrawFilterCollaboratorsArea();

            GUILayout.Space(10);

            Rect treeRect = GUILayoutUtility.GetRect(
                0,
                0,
                GUILayout.MaxWidth(LAYOUT_MAX_WIDTH),
                GUILayout.MaxHeight(300));
            mCollaboratorsListView.OnGUI(treeRect);

            GUILayout.Space(5);

            DrawManageUsersLink(isOperationRunning);

            GUILayout.Space(5);
        }

        void DrawFilterCollaboratorsArea()
        {
            GUILayout.BeginHorizontal(GUILayout.MaxWidth(LAYOUT_MAX_WIDTH));
            GUILayout.FlexibleSpace();

            DrawSearchField.For(
                mSearchField,
                mCollaboratorsListView,
                UnityConstants.SEARCH_FIELD_WIDTH);

            GUILayout.EndHorizontal();
        }

        void DrawManageUsersLink(bool isOperationRunning)
        {
            GUILayout.BeginHorizontal();

            if (DoButton(
                    PlasticLocalization.Name.ManageUsersDashboardLink.GetString(),
                    UnityStyles.LinkLabel,
                    !isOperationRunning,
                    ENTRY_WIDTH))
            {
                OpenManageUsersUrl(mSelectedOrganization, mSelectedProject);
            }

            GUILayout.FlexibleSpace();
            GUILayout.EndHorizontal();
        }

        void FocusWorkspaceNameEntryIfNeeded()
        {
            if (mWasWorkspaceNameFocused)
                return;

            EditorGUI.FocusTextInControl(WKNAME_CONTROL_NAME);

            mWasWorkspaceNameFocused = true;
        }

        void CreateWorkspaceButtonValidationAction()
        {
            mWkCreationData = BuildCreationData();

            // It calls IPlasticDialogCloser.CloseDialog() when the validation is OK
            WorkspaceCreationValidation.AsyncValidation(
                mWkCreationData, this, mProgressControls);
        }

        WorkspaceCreationData BuildCreationData()
        {
            string name = CloudServer.IsUnityOrganization(mSelectedOrganization) ?
                CloudProjectRepository.BuildFullyQualifiedName(
                    mSelectedProject, mWorkspaceName) :
                mWorkspaceName;

            return new WorkspaceCreationData(
                mSelectedOrganization,
                name,
                mCollaboratorsListView.GetCollaborators());
        }

        static void OpenOrganizationProjectUrl(string selectedOrganization)
        {
            string solvedServer = ResolveServer.FromUserInput(
                selectedOrganization, CmConnection.Get().UnityOrgResolver);

            string genesisOrgId;
            if (!CloudServer.TryGetGenesisOrgId(solvedServer, out genesisOrgId))
                return;

            Application.OpenURL(UnityUrl.UnityDashboard.
                UnityOrganizations.GetProjectsUrl(genesisOrgId));
        }

        static void OpenManageUsersUrl(
            string selectedOrganization,
            string selectedProject)
        {
            string solvedServer = ResolveServer.FromUserInput(
                selectedOrganization, CmConnection.Get().UnityOrgResolver);

            if (!CloudServer.IsUnityOrganization(selectedOrganization))
            {
                Application.OpenURL(
                    UnityUrl.UnityDashboard.Plastic.GetForManageUsersLegacyOrg(
                        CloudServer.GetOrganizationName(selectedOrganization)));
                return;
            }

            string genesisOrgId;
            if (!CloudServer.TryGetGenesisOrgId(solvedServer, out genesisOrgId))
                return;

            Application.OpenURL(
                UnityUrl.UnityDashboard.UnityOrganizations.GetManageMembersUrl(
                    genesisOrgId,
                    GetProjectGuid(selectedOrganization, selectedProject)));
        }

        static string GetProjectGuid(string organization, string projectName)
        {
            if (!CloudServer.IsUnityOrganization(organization))
                return string.Empty;

            return PlasticGui.Plastic.API.GetRepositoryInfo(
                RepositorySpec.BuildFromNameAndResolvedServer(
                    projectName, organization)).GUID.ToString();
        }

        static void DoTitleArea(string explanation)
        {
            GUILayout.Label(PlasticLocalization.Name.CreateCloudWorkspaceTitle.GetString(),
                UnityStyles.Dialog.Title);

            GUILayout.Space(5);

            GUILayout.Label(explanation, UnityStyles.Paragraph);

            GUILayout.Space(10);
        }

        static void DoCreateOrganizationProjectLink(
            string selectedOrganization,
            bool isOperationRunning)
        {
            string buttonText = PlasticLocalization.Name.
                CreateOrganizationProjectLabel.GetString();

            if (DoButton(
                    buttonText,
                    UnityStyles.LinkLabel,
                    !isOperationRunning,
                    ENTRY_WIDTH,
                    ENTRY_X + ENTRY_WIDTH - PROJECT_LINK_WIDTH))
            {
                OpenOrganizationProjectUrl(selectedOrganization);
            }
        }

        static void DoCreateWorkspaceArea(
            bool isInputValid,
            ProgressControlsForViews progressControls,
            Action createWorkspaceButtonValidationAction)
        {
            Rect rect = EditorGUILayout.BeginHorizontal(GUILayout.Height(24));

            bool isButtonEnabled = isInputValid && !progressControls.IsOperationRunning();

            bool isButtonClicked = DoButton(
                PlasticLocalization.Name.CreateCloudWorkspaceButton.GetString(),
                GUI.skin.button,
                isButtonEnabled,
                CREATE_WORKSPACE_BUTTON_WIDTH,
                CREATE_WORKSPACE_BUTTON_MARGIN,
                rect.y + 2);

            GUILayout.Space(5);

            if (progressControls.IsOperationRunning())
                DoProgress(progressControls.ProgressData);

            EditorGUILayout.EndHorizontal();

            if (isButtonClicked)
                createWorkspaceButtonValidationAction();
        }

        static bool DoButton(
            string text,
            GUIStyle style,
            bool isEnabled,
            float width,
            float x = -1,
            float y = -1)
        {
            using (new GuiEnabled(isEnabled))
            {
                GUIContent buttonContent = new GUIContent(text);

                Rect rect = GUILayoutUtility.GetRect(
                    buttonContent, style,
                    GUILayout.MinWidth(width),
                    GUILayout.MaxWidth(width));

                if (x != -1)
                    rect.x = x;

                if (y != -1)
                    rect.y = y;

                bool result = GUI.Button(rect, buttonContent, style);

                return result;
            }
        }

        static void DoWarningLabel(
            string text,
            float x)
        {
            Rect rect = GUILayoutUtility.GetRect(
                new GUIContent(text),
                EditorStyles.label);
            rect.x = x;

            GUI.Label(rect,
                new GUIContent(text, Images.GetWarnIcon()),
                UnityStyles.HeaderWarningLabel);
        }

        static void DoProgress(
            ProgressControlsForViews.Data data)
        {
            if (string.IsNullOrEmpty(data.ProgressMessage))
                return;

            DrawProgressForViews.ForIndeterminateProgressSpinner(data);
        }

        static void DoNotificationArea(
            ProgressControlsForViews.Data data)
        {
            if (string.IsNullOrEmpty(data.NotificationMessage))
                return;

            DrawProgressForViews.ForNotificationArea(data);
        }

        static InputValidationResult ValidateWorkspaceNameInput(string workspaceName)
        {
            InputValidationResult result = new InputValidationResult();

            result.IsValid = InputValidator.IsValidRepositoryName(
                workspaceName,
                PlasticLocalization.Name.CloudWorkspaceNameEmpty.GetString(),
                out result.ErrorMessage);

            return result;
        }

        static string GetDefaultValue(string proposedValue, List<string> values)
        {
            if (values.Count == 0)
                return null;

            if (proposedValue != null && values.Contains(proposedValue))
                return proposedValue;

            return values[0];
        }

        static string GetExplanationText(string selectedOrganization)
        {
            return PlasticLocalization.Name.CreateCloudWorkspaceExplanation.GetString(
                CloudServer.IsUnityOrganization(selectedOrganization) ?
                    PlasticLocalization.Name.CreateCloudWorkspaceExplanationSelectForUnityOrg.GetString() :
                    PlasticLocalization.Name.CreateCloudWorkspaceExplanationSelectForCloudOrg.GetString());
        }

        static bool IsInputValid(
            string selectedOrganization,
            string selectedProject,
            string workspaceName)
        {
            if (string.IsNullOrEmpty(selectedOrganization))
                return false;

            if (CloudServer.IsUnityOrganization(selectedOrganization) &&
                string.IsNullOrEmpty(selectedProject))
                return false;

            return ValidateWorkspaceNameInput(workspaceName).IsValid;
        }

        void BuildComponents(ProgressControlsForViews progressControls)
        {
            mExplanation = PlasticLocalization.Name.CreateCloudWorkspaceExplanation.GetString(
                PlasticLocalization.Name.CreateCloudWorkspaceExplanationSelectForUnityOrg.GetString());

            GetProposedOrganizationProject.Values proposedOrganizationProject =
                GetProposedOrganizationProject.FromCloudProjectSettings();

            if (proposedOrganizationProject != null)
            {
                mProposedOrganization = proposedOrganizationProject.Organization;
                mProposedProject = proposedOrganizationProject.Project;
            }

            mProposedWorkspaceName = Application.productName;

            CollaboratorsListViewHeaderState collaboratorsListHeaderState =
                CollaboratorsListViewHeaderState.GetDefault();
            TreeHeaderSettings.Load(
                collaboratorsListHeaderState,
                UnityConstants.CloudDrive.COLLABORATORS_TABLE_SETTINGS_NAME,
                (int)CollaboratorsListColumn.User,
                true);

            mSearchField = new SearchField();
            mCollaboratorsListView = new CollaboratorsListView(
                progressControls, collaboratorsListHeaderState, mParentWindow.Repaint);
            mCollaboratorsListView.Reload();
        }

        class InputValidationResult
        {
            internal string ErrorMessage;
            internal bool IsValid;
        }

        WorkspaceCreationData mWkCreationData;

        bool mWasWorkspaceNameFocused;
        string mExplanation;
        string mProposedOrganization;
        string mProposedProject;
        string mProposedWorkspaceName;
        string mSelectedOrganization;
        string mSelectedProject;
        string mWorkspaceName;

        List<string> mOrganizations = new List<string>();
        List<string> mProjects = new List<string>();

        SearchField mSearchField;
        CollaboratorsListView mCollaboratorsListView;

        readonly EditorWindow mParentWindow;
        readonly ProgressControlsForViews mProgressControls;
        readonly ICreateWorkspaceListener mCreateWorkspaceListener;
        readonly IPlasticAPI mPlasticApi;

        const float PROJECT_LINK_WIDTH = 192;
        const float CREATE_WORKSPACE_BUTTON_MARGIN = 32;
        const float CREATE_WORKSPACE_BUTTON_WIDTH = 160;
        const float ENTRY_WIDTH = 400;
        const float ENTRY_X = 175f;
        const float LAYOUT_MAX_WIDTH = 800;

        const string WKNAME_CONTROL_NAME = "CreateWorkspaceView.WorkspaceNameTextField";
    }
}
