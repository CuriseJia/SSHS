using UnityEditor;
using UnityEngine;

using Codice.Client.Common.WebApi;
using PlasticGui;

namespace Unity.PlasticSCM.Editor.CloudDrive.CreateWorkspace.Welcome
{
    internal class WelcomeView
    {
        internal WelcomeView(
            IPlasticWebRestApi restApi,
            IPlasticAPI plasticApi,
            CreateWorkspaceView.ICreateWorkspaceListener listener,
            EditorWindow parentWindow)
        {
            BuildComponents(restApi, plasticApi, listener, parentWindow);
        }

        internal void Update()
        {
            mCreateWorkspaceView.Update();
        }

        internal void OnGUI(Rect rect)
        {
            GUILayout.BeginArea(rect);

            GUILayout.BeginHorizontal();

            GUILayout.Space(HORIZONTAL_MARGIN);

            DoContentViewArea();

            GUILayout.Space(HORIZONTAL_MARGIN);

            GUILayout.EndHorizontal();

            GUILayout.EndArea();
        }

        void DoContentViewArea()
        {
            GUILayout.BeginVertical();

            GUILayout.Space(TOP_MARGIN);

            mCreateWorkspaceView.OnGUI();

            GUILayout.EndVertical();
        }

        void BuildComponents(
            IPlasticWebRestApi restApi,
            IPlasticAPI plasticApi,
            CreateWorkspaceView.ICreateWorkspaceListener listener,
            EditorWindow parentWindow)
        {
            mCreateWorkspaceView = new CreateWorkspaceView(
                restApi, plasticApi, listener, parentWindow);
        }

        CreateWorkspaceView mCreateWorkspaceView;

        const int HORIZONTAL_MARGIN = 30;
        const int TOP_MARGIN = 20;
    }
}
