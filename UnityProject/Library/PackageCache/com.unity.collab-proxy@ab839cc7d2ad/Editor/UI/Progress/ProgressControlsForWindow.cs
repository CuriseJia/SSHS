using System;

using UnityEditor;

using PlasticGui;

namespace Unity.PlasticSCM.Editor.UI.Progress
{
    internal class ProgressControlsForWindow : IProgressControls
    {
        internal class Data
        {
            internal bool IsOperationRunning;
            internal float ProgressPercent;
            internal string ProgressMessage;
        }

        internal Data ProgressData { get { return mData; } }

        internal void UpdateProgress(EditorWindow parentWindow)
        {
            if (IsOperationRunning() || mRequestedRepaint)
            {
                if (IsOperationRunning())
                    UpdateIndeterminateProgress();

                parentWindow.Repaint();

                mRequestedRepaint = false;
            }
        }

        void IProgressControls.HideProgress()
        {
            mData.IsOperationRunning = false;
            mData.ProgressMessage = string.Empty;

            mRequestedRepaint = true;
        }

        void IProgressControls.ShowProgress(string message)
        {
            mData.IsOperationRunning = true;
            mData.ProgressMessage = message;

            mRequestedRepaint = true;
        }

        void IProgressControls.ShowError(string message)
        {
            throw new NotImplementedException();
        }

        void IProgressControls.ShowNotification(string message)
        {
            throw new NotImplementedException();
        }

        void IProgressControls.ShowSuccess(string message)
        {
            throw new NotImplementedException();
        }

        void IProgressControls.ShowWarning(string message)
        {
            throw new NotImplementedException();
        }

        bool IsOperationRunning()
        {
            return mData.IsOperationRunning;
        }

        void UpdateIndeterminateProgress()
        {
            mData.ProgressPercent += .01f;

            if (mData.ProgressPercent > 1f)
                mData.ProgressPercent = 0f;
        }

        Data mData = new Data();

        bool mRequestedRepaint;
    }
}
