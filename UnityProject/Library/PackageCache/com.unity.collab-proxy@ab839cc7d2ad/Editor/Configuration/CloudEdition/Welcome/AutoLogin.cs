using System;
using UnityEditor;
using UnityEngine;

using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.LogWrapper;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.WebApi;

namespace Unity.PlasticSCM.Editor.Configuration.CloudEdition.Welcome
{
    internal class AutoLogin
    {
        internal enum State : byte
        {
            Off = 0,
            Started = 1,
            Running = 2,
            ResponseInit = 3,
            ResponseEnd = 4,
            ResponseSuccess = 5,
            ErrorNoToken = 20,
            ErrorTokenException = 21,
            ErrorResponseNull = 22,
            ErrorResponseError = 23,
            ErrorTokenEmpty = 24,
        }

        internal bool Run()
        {
            mLog.Debug("Run");

            mUVCSWindow = GetUVCSWindow();

            if (string.IsNullOrEmpty(CloudProjectSettings.accessToken))
            {
                mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorNoToken;
                return false;
            }

            mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.Running;

            ExchangeTokensAndJoinOrganizationInThreadWaiter(CloudProjectSettings.accessToken);

            return true;
        }

        void ExchangeTokensAndJoinOrganizationInThreadWaiter(string unityAccessToken)
        {
            int ini = Environment.TickCount;

            TokenExchangeResponse tokenExchangeResponse = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter(10);
            waiter.Execute(
            /*threadOperationDelegate*/ delegate
            {
                mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ResponseInit;
                tokenExchangeResponse = WebRestApiClient.PlasticScm.TokenExchange(unityAccessToken);
            },
            /*afterOperationDelegate*/ delegate
            {
                mLog.DebugFormat(
                    "TokenExchange time {0} ms",
                    Environment.TickCount - ini);

                if (waiter.Exception != null)
                {
                    mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorTokenException;
                    ExceptionsHandler.LogException(
                        "TokenExchangeSetting",
                        waiter.Exception);
                    Debug.LogWarning(waiter.Exception.Message);
                    return;
                }

                if (tokenExchangeResponse == null)
                {
                    mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorResponseNull;
                    var warning = PlasticLocalization.GetString(PlasticLocalization.Name.TokenExchangeResponseNull);
                    mLog.Warn(warning);
                    Debug.LogWarning(warning);
                    return;
                }

                if (tokenExchangeResponse.Error != null)
                {
                    mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorResponseError;
                    var warning = string.Format(
                        PlasticLocalization.GetString(PlasticLocalization.Name.TokenExchangeResponseError),
                        tokenExchangeResponse.Error.Message, tokenExchangeResponse.Error.ErrorCode);
                    mLog.ErrorFormat(warning);
                    Debug.LogWarning(warning);
                    return;
                }

                if (string.IsNullOrEmpty(tokenExchangeResponse.AccessToken))
                {
                    mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ErrorTokenEmpty;
                    var warning = string.Format(
                        PlasticLocalization.GetString(PlasticLocalization.Name.TokenExchangeAccessEmpty),
                        tokenExchangeResponse.User);
                    mLog.InfoFormat(warning);
                    Debug.LogWarning(warning);
                    return;
                }

                mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ResponseEnd;

                Credentials credentials = new Credentials(
                    new SEID(tokenExchangeResponse.User, false, tokenExchangeResponse.AccessToken),
                    SEIDWorkingMode.SSOWorkingMode);

                ShowOrganizationsPanel(credentials);
            });
        }

        void ShowOrganizationsPanel(Credentials credentials)
        {
            mUVCSWindow = GetUVCSWindow();
            mUVCSWindow.GetWelcomeView().autoLoginState = AutoLogin.State.ResponseSuccess;

            CloudEditionWelcomeWindow.ShowWindow(
                PlasticGui.Plastic.WebRestAPI, null, true);

            mCloudEditionWelcomeWindow = CloudEditionWelcomeWindow.GetWelcomeWindow();

            mCloudEditionWelcomeWindow.GetOrganizations(credentials);

            mCloudEditionWelcomeWindow.Focus();
        }

        static UVCSWindow GetUVCSWindow()
        {
            var windows = Resources.FindObjectsOfTypeAll<UVCSWindow>();
            UVCSWindow uvcsWindow = windows.Length > 0 ? windows[0] : null;

            if (uvcsWindow == null)
                uvcsWindow = ShowWindow.UVCS();

            return uvcsWindow;
        }

        UVCSWindow mUVCSWindow;
        CloudEditionWelcomeWindow mCloudEditionWelcomeWindow;

        static readonly ILog mLog = PlasticApp.GetLogger("AutoLogin");
    }
}
