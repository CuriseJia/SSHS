using UnityEditor.PackageManager.UI;
using UnityEngine;

namespace Unity.PlasticSCM.Editor
{
    internal static class LaunchPackageManager
    {
        internal static void Open(string packageName)
        {
            Window.Open(packageName);
        }

        internal static void AddByName(string packageName, string packageVersion)
        {
            Open(packageName);

            const string upmUrl = "com.unity3d.kharma:upmpackage/";
            string url = string.Format("{0}{1}@{2}", upmUrl, packageName, packageVersion);
            Application.OpenURL(url);
        }
    }
}
