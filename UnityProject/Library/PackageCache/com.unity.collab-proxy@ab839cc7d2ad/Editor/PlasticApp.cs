using System;
using System.IO;
using System.Linq;

using UnityEditor;
using UnityEngine;

using Codice.Client.BaseCommands;
using Codice.Client.Common;
using Codice.Client.Common.Connection;
using Codice.Client.Common.Encryption;
using Codice.Client.Common.EventTracking;
using Codice.Client.Common.Threading;
using Codice.Client.Common.WebApi;
using Codice.CM.Common;
using Codice.CM.ConfigureHelper;
using Codice.LogWrapper;
using CodiceApp.EventTracking;
using PlasticGui;
using PlasticPipe.Certificates;
using PlasticPipe.Client;
using Unity.PlasticSCM.Editor.CloudDrive;
using Unity.PlasticSCM.Editor.Configuration;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor
{
    internal static class PlasticApp
    {
        internal static bool IsUnitTesting { get; set; }

        internal static ILog GetLogger(string name)
        {
            if (!mIsDomainUnloadHandlerRegistered)
            {
                // Register the Domain Unload Handler before the LogManager is initialized,
                // so the domain unload handler for the app is processed before the log manager one,
                // and thus the AppDomainUnload is printed in the log
                RegisterDomainUnloadHandler();
                mIsDomainUnloadHandlerRegistered = true;
            }

            return LogManager.GetLogger(name);
        }

        internal static void InitializeIfNeeded()
        {
            if (mIsInitialized)
                return;

            mIsInitialized = true;

            // Configure logging on initialize to avoid adding the performance cost of it
            // on every Editor load and Domain reload for non-UVCS users.
            ConfigureLogging();

            mLog.Debug("InitializeIfNeeded");

            mLog.DebugFormat("Unity version: {0}", Application.unityVersion);
            mLog.DebugFormat("unityplastic.dll version: {0}", UnityPlasticDllVersion.GetFileVersion());

            // Ensures that the Edition Token is initialized
            UnityConfigurationChecker.SynchronizeUnityEditionToken();
            PlasticInstallPath.LogInstallationInfo();

            if (!IsUnitTesting)
                GuiMessage.Initialize(new UnityPlasticGuiMessage());

            RegisterExceptionHandlers();
            RegisterApplicationFocusHandlers();
            RegisterBeforeAssemblyReloadHandler();
            RegisterEditorWantsToQuit();
            RegisterEditorQuitting();

            InitLocalization();

            if (!IsUnitTesting)
                ThreadWaiter.Initialize(new UnityThreadWaiterBuilder());

            ServicePointConfigurator.ConfigureServicePoint();
            CertificateUi.RegisterHandler(new ChannelCertificateUiImpl());

            EditionManager.Get().DisableCapability(EnumEditionCapabilities.Extensions);

            ClientHandlers.Register();

            PlasticGuiConfig.SetConfigFile(PlasticGuiConfig.UNITY_GUI_CONFIG_FILE);

            if (!IsUnitTesting)
            {
                mEventSenderScheduler = EventTracking.Configure(
                    (PlasticWebRestApi)PlasticGui.Plastic.WebRestAPI,
                    ApplicationIdentifier.UnityPackage,
                    IdentifyEventPlatform.Get());
            }

            PackageInfo.Initialize();

            PlasticMethodExceptionHandling.InitializeAskCredentialsUi(
                new CredentialsUiImpl());
            ClientEncryptionServiceProvider.SetEncryptionPasswordProvider(
                new MissingEncryptionPasswordPromptHandler());
        }

        internal static void Enable()
        {
            string webApiToken = CmConnection.Get().BuildWebApiTokenForCloudEditionDefaultUser();

            if (string.IsNullOrEmpty(webApiToken))
                return;

            PlasticGui.Plastic.WebRestAPI.SetToken(webApiToken);

            OrganizationsInformation.UpdateCloudOrganizationSlugsAsync(
                PlasticGui.Plastic.WebRestAPI, PlasticGui.Plastic.API);
        }

        internal static void DisposeIfNeeded()
        {
            if (!mIsInitialized)
                return;

            if (UVCSPlugin.Instance.IsEnabled() ||
                CloudDrivePlugin.Instance.IsEnabled())
                return;

            mLog.Debug("Dispose");

            mIsInitialized = false;

            UnRegisterDomainUnloadHandler();
            UnRegisterExceptionHandlers();
            UnRegisterApplicationFocusHandlers();
            UnRegisterBeforeAssemblyReloadHandler();
            UnRegisterEditorWantsToQuit();
            UnRegisterEditorQuitting();

            if (mEventSenderScheduler != null)
            {
                // Launching and forgetting to avoid a timeout when sending events files and no
                // network connection is available.
                // This will be refactored once a better mechanism to send event is in place
                mEventSenderScheduler.EndAndSendEventsAsync();
            }

            ClientConnectionPool.Shutdown();
        }

        static void Shutdown()
        {
            UVCSPlugin.Instance.Shutdown();

            CloudDrivePlugin.Instance.Shutdown();

            DisposeIfNeeded();
        }

        static void RegisterDomainUnloadHandler()
        {
            AppDomain.CurrentDomain.DomainUnload += AppDomainUnload;
        }

        static void RegisterExceptionHandlers()
        {
            AppDomain.CurrentDomain.UnhandledException += HandleUnhandledException;

            Application.logMessageReceivedThreaded += HandleLog;
        }

        static void RegisterApplicationFocusHandlers()
        {
            EditorWindowFocus.OnApplicationActivated += OnApplicationActivated;

            EditorWindowFocus.OnApplicationDeactivated += OnApplicationDeactivated;
        }

        static void RegisterBeforeAssemblyReloadHandler()
        {
            AssemblyReloadEvents.beforeAssemblyReload += BeforeAssemblyReload;
        }

        static void RegisterEditorWantsToQuit()
        {
            EditorApplication.wantsToQuit += OnEditorWantsToQuit;
        }

        static void RegisterEditorQuitting()
        {
            EditorApplication.quitting += OnEditorQuitting;
        }

        static void UnRegisterDomainUnloadHandler()
        {
            AppDomain.CurrentDomain.DomainUnload -= AppDomainUnload;
        }

        static void UnRegisterExceptionHandlers()
        {
            AppDomain.CurrentDomain.UnhandledException -= HandleUnhandledException;

            Application.logMessageReceivedThreaded -= HandleLog;
        }

        static void UnRegisterApplicationFocusHandlers()
        {
            EditorWindowFocus.OnApplicationActivated -= OnApplicationActivated;

            EditorWindowFocus.OnApplicationDeactivated -= OnApplicationDeactivated;
        }

        static void UnRegisterBeforeAssemblyReloadHandler()
        {
            AssemblyReloadEvents.beforeAssemblyReload -= BeforeAssemblyReload;
        }

        static void UnRegisterEditorWantsToQuit()
        {
            EditorApplication.wantsToQuit -= OnEditorWantsToQuit;
        }

        static void UnRegisterEditorQuitting()
        {
            EditorApplication.quitting -= OnEditorQuitting;
        }

        static void AppDomainUnload(object sender, EventArgs e)
        {
            mLog.Debug("AppDomainUnload");

            UnRegisterDomainUnloadHandler();
        }

        static void HandleUnhandledException(object sender, UnhandledExceptionEventArgs args)
        {
            Exception ex = (Exception)args.ExceptionObject;

            if (CheckUnityException.IsExitGUIException(ex) ||
                !IsPlasticStackTrace(ex.StackTrace))
                throw ex;

            GUIActionRunner.RunGUIAction(delegate {
                ExceptionsHandler.HandleException("HandleUnhandledException", ex);
            });
        }

        static void HandleLog(string logString, string stackTrace, LogType type)
        {
            if (type != LogType.Exception)
                return;

            if (!IsPlasticStackTrace(stackTrace))
                return;

            GUIActionRunner.RunGUIAction(delegate {
                mLog.ErrorFormat("[HandleLog] Unexpected error: {0}", logString);
                mLog.DebugFormat("Stack trace: {0}", stackTrace);

                string message = logString;
                if (ExceptionsHandler.DumpStackTrace())
                    message += Environment.NewLine + stackTrace;

                GuiMessage.ShowError(message);
            });
        }

        static void OnApplicationActivated()
        {
            mLog.Debug("OnApplicationActivated");

            if (UVCSPlugin.Instance.IsEnabled())
                UVCSPlugin.Instance.OnApplicationActivated();

            if (CloudDrivePlugin.Instance.IsEnabled())
                CloudDrivePlugin.Instance.OnApplicationActivated();
        }

        static void OnApplicationDeactivated()
        {
            mLog.Debug("OnApplicationDeactivated");

            if (UVCSPlugin.Instance.IsEnabled())
                UVCSPlugin.Instance.OnApplicationDeactivated();
        }

        static void BeforeAssemblyReload()
        {
            mLog.Debug("BeforeAssemblyReload");

            UnRegisterBeforeAssemblyReloadHandler();

            Shutdown();
        }

        static bool OnEditorWantsToQuit()
        {
            mLog.Debug("OnEditorWantsToQuit");

            return UVCSPlugin.Instance.OnEditorWantsToQuit()
                && CloudDrivePlugin.Instance.OnEditorWantsToQuit();
        }

        static void OnEditorQuitting()
        {
            mLog.Debug("OnEditorQuitting");

            Shutdown();
        }

        static void ConfigureLogging()
        {
            try
            {
                string log4netpath = ToolConfig.GetUnityPlasticLogConfigFile();

                if (!File.Exists(log4netpath))
                    WriteLogConfiguration.For(log4netpath);

                XmlConfigurator.Configure(new FileInfo(log4netpath));

                mLog.DebugFormat("Configured logging in '{0}'", log4netpath);
            }
            catch
            {
                //it failed configuring the logging info; nothing to do.
            }
        }

        static void InitLocalization()
        {
            string language = null;
            try
            {
                language = ClientConfig.Get().GetLanguage();
            }
            catch
            {
                language = string.Empty;
            }

            Localization.Init(language);
            PlasticLocalization.SetLanguage(language);
        }

        static bool IsPlasticStackTrace(string stackTrace)
        {
            if (stackTrace == null)
                return false;

            string[] namespaces = new[] {
                "Codice.",
                "GluonGui.",
                "PlasticGui."
            };

            return namespaces.Any(stackTrace.Contains);
        }

        static bool mIsDomainUnloadHandlerRegistered;
        static bool mIsInitialized;
        static EventSenderScheduler mEventSenderScheduler;

        static readonly ILog mLog = PlasticApp.GetLogger("PlasticApp");
    }
}
