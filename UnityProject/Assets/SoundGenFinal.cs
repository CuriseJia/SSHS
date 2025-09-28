using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using System.IO;
using Newtonsoft.Json;
using SteamAudioSource = SteamAudio.SteamAudioSource;

[RequireComponent(typeof(AudioListener))]
public class SoundGenFinal : MonoBehaviour
{
    private AudioClip loadedClip = null;
    string filePath = "C:\\Users\\zhanglab\\Desktop\\audio\\test";
    string exportPath = "E:\\NeurIPSAudio";
    private List<float> recordedSamples = new List<float>();
    public int outputSampleRate = 44100;
    public int outputChannels = 2;
    private AudioClip audioClip = null;
    private AudioClip targetAudioClip = null;
    private AudioClip distAudioClip = null;
    private bool recording = false;

    [Serializable]
    public class Target
    {
        public string image_id { get; set; }
        public List<float> coords { get; set; }
        public string audio { get; set; }
        public string output { get; set; }
    }


    private IEnumerator Start()
    {
        TextAsset[] jsonFiles = Resources.LoadAll<TextAsset>("configFiles");

        foreach (TextAsset jsonFile in jsonFiles)
        {
            List<Target> targets = JsonConvert.DeserializeObject<List<Target>>(jsonFile.text);

            foreach (var target in targets)
            {
                List<float> coordinates = target.coords;

                string targetAudioImport = filePath + "/" + target.audio;
                Vector3 targetPos = new Vector3(coordinates[0] / 1920f * 224f, coordinates[1] / 1080f * 224f, coordinates[2]);
                string finalExportPath = exportPath + "/" + target.output;
                if (File.Exists(finalExportPath))
                {
                    continue;
                }
                yield return StartCoroutine(AudioGeneration(targetAudioImport, finalExportPath, targetPos));
            }
        }

        Debug.Log("Experiment Completed");
        yield break;
    }

    private IEnumerator AudioGeneration(string filePath, string exportPath, Vector3 position)
    {
        bool clipLoaded = false;

        yield return StartCoroutine(LoadAudioFromFile(filePath, (clip) =>
        {
            audioClip = clip;
            clipLoaded = true;
        }));

        if (audioClip != null)
        {
            PlaceAudioAtLocation(audioClip, position);

            recordedSamples.Clear();
            recording = true;

            yield return new WaitForSeconds(audioClip.length);

            recording = false;

            byte[] wavData = ConvertToWav(recordedSamples.ToArray(), outputChannels, outputSampleRate);

            string saveFolder = Path.GetDirectoryName(exportPath);
            if (!string.IsNullOrEmpty(saveFolder) && !Directory.Exists(saveFolder))
            {
                Directory.CreateDirectory(saveFolder);
            }

            File.WriteAllBytes(exportPath, wavData);
            Debug.Log("Exported WAV to: " + exportPath);
            recordedSamples.Clear();
        }
    }

    private void PlaceAudioAtLocation(AudioClip clip, Vector3 pos)
    {
        GameObject audioObj = new GameObject("SoundObj");
        audioObj.transform.position = pos;

        AudioSource audio = audioObj.AddComponent<AudioSource>();
        audio.clip = clip;
        audio.spatialBlend = 1.0f;
        audio.spatialize = true;
        audio.volume = 1.0f;
        audio.loop = false;
        audio.rolloffMode = AudioRolloffMode.Logarithmic;
        audio.minDistance = 0.76f;
        audio.maxDistance = 1.5f;

        SteamAudioSource steamAudioSource = audioObj.AddComponent<SteamAudioSource>();
        steamAudioSource.directBinaural = true;
        steamAudioSource.distanceAttenuation = true;

        audio.Play();
        Destroy(audioObj, clip.length);
    }

    private IEnumerator LoadAudioFromFile(string path, System.Action<AudioClip> onLoaded)
    {
        UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(path, AudioType.WAV);
        yield return www.SendWebRequest();

        if (www.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"Failed to load audio from '{path}' due to {www.error}");
            onLoaded?.Invoke(null);
        }
        else
        {
            Debug.Log($"Loading audio from '{path}'");
            try
            {
                AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
                Debug.Log($"Successfully loaded external clip of Length: {clip.length} seconds, Channels: {clip.channels}");
                onLoaded?.Invoke(clip);
            }
            catch
            {
                Debug.Log($"Unable to load audio from '{path}'");
                onLoaded?.Invoke(null);
            }
        }
    }

    private void OnAudioFilterRead(float[] data, int channels)
    {
        if (!recording) return;
        recordedSamples.AddRange(data);
    }

    private byte[] ConvertToWav(float[] samples, int channels, int sampleRate)
    {
        int sampleCount = samples.Length;
        int byteCount = sampleCount * sizeof(short);
        int headerSize = 44;
        int fileSize = headerSize + byteCount;

        using (MemoryStream memStream = new MemoryStream(fileSize))
        using (BinaryWriter writer = new BinaryWriter(memStream))
        {
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(fileSize - 8);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            writer.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(sampleRate * channels * sizeof(short));
            writer.Write((short)(channels * sizeof(short)));
            writer.Write((short)16);

            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(byteCount);

            for (int i = 0; i < sampleCount; i++)
            {
                float f = Mathf.Clamp(samples[i], -1f, 1f);
                short s = (short)(f * short.MaxValue);
                writer.Write(s);
            }
            return memStream.ToArray();
        }
    }
}