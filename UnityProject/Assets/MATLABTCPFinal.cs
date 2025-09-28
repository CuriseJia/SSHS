using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using UnityEngine.Networking;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using System.Threading;
using SteamAudioSource = SteamAudio.SteamAudioSource;

[RequireComponent(typeof(AudioListener))]
public class MATLABTCPFinal : MonoBehaviour
{
    private ConcurrentQueue<System.Action> mainThreadActions = new ConcurrentQueue<System.Action>();
    private Thread listenerThread;
    private TcpListener listener;
    private TcpClient client;
    private NetworkStream stream;
    private StreamReader reader;
    private StreamWriter writer;
    private AudioClip audioClip = null;
    private AudioClip targetAudioClip = null;
    private AudioClip distAudioClip = null;
    private GameObject curAudioObj = null;
    private List<GameObject> audioObjects = new List<GameObject>();

    private class JsonData
    {
        public string cat;
        public string data;
    }

    public class Coordinates
    {
        public float x;
        public float y;
        public float z;
    }

    private class AudioData
    {
        public bool getCoords;
        public bool start;
        public float[][] coords;
        public string[] audio;
    }

    void Start()
    {
        listener = new TcpListener(IPAddress.Parse("127.0.0.1"), 55001);
        listener.Start();

        listenerThread = new Thread(MatlabListener);
        listenerThread.IsBackground = true;
        listenerThread.Start();
    }

    private void MatlabListener()
    {
        client = listener.AcceptTcpClient();
        stream = client.GetStream();
        reader = new StreamReader(stream, Encoding.UTF8);
        writer = new StreamWriter(stream, new UTF8Encoding(false));

        while (true)
        {
            string json = reader.ReadLine();
            if (json == null)
            {
                Debug.Log("Matlab has disconnected");
                break;
            }
            Debug.Log("Received JSON: " + json);
            mainThreadActions.Enqueue(() => JsonHandler(json));
            Thread.Sleep(10);
        }

        reader?.Close();
        writer?.Close();
        stream?.Close();
        client?.Close();
    }

    void Update()
    {
        while (mainThreadActions.Count > 0)
        {
            if (mainThreadActions.TryDequeue(out System.Action action))
            {
                action?.Invoke();
            }
        }
    }

    private void JsonHandler(string jsonData)
    {
        JsonData unpackedData = JsonConvert.DeserializeObject<JsonData>(jsonData);
        string cat = unpackedData.cat;
        string data = unpackedData.data;

        switch (cat)
        {
            case "start":
                StartCoroutine(loadAudio(data));
                break;

            case "end":
                foreach (GameObject audioObj in audioObjects)
                {
                    if (audioObj != null)
                    {
                        Destroy(audioObj);
                    }
                }
                audioObjects.Clear();
                break;

            case "get":
                curAudioObj = audioObjects[0];
                Vector3 position = curAudioObj.transform.position;

                Coordinates coords = new Coordinates
                {
                    x = position.x,
                    y = position.y,
                    z = position.z
                };
                string jsonSend = JsonConvert.SerializeObject(coords);

                writer.WriteLine(jsonSend);
                writer.Flush();
                Debug.Log("Sent Coordinates to Matlab");
                break;

            case "move":
                curAudioObj = audioObjects[0];

                float UP = 3f; float DOWN = -3f;
                float LEFT = -3f; float RIGHT = 3f;
                float IN = 3f; float OUT = -3f;

                switch (data)
                {
                    case "W":
                        curAudioObj.transform.Translate(0f, UP * Time.deltaTime, 0f);
                        break;
                    case "A":
                        curAudioObj.transform.Translate(LEFT * Time.deltaTime, 0f, 0f);
                        break;
                    case "S":
                        curAudioObj.transform.Translate(0f, DOWN * Time.deltaTime, 0f);
                        break;
                    case "D":
                        curAudioObj.transform.Translate(RIGHT * Time.deltaTime, 0f, 0f);
                        break;
                    case "WA":
                        curAudioObj.transform.Translate(LEFT * Time.deltaTime, UP * Time.deltaTime, 0f);
                        break;
                    case "WD":
                        curAudioObj.transform.Translate(RIGHT * Time.deltaTime, UP * Time.deltaTime, 0f);
                        break;
                    case "SA":
                        curAudioObj.transform.Translate(LEFT * Time.deltaTime, DOWN * Time.deltaTime, 0f);
                        break;
                    case "SD":
                        curAudioObj.transform.Translate(RIGHT * Time.deltaTime, DOWN * Time.deltaTime, 0f);
                        break;
                    case "0":
                        curAudioObj.transform.Translate(0f, 0f, 0f);
                        break;
                }
                break;
        }
    }

    private IEnumerator loadAudio(string jsonData)
    {
        AudioData data = JsonConvert.DeserializeObject<AudioData>(jsonData);
        if (data != null)
        {
            if (data.audio.Length == 1)
            {
                Vector3 targetPosition = new Vector3(data.coords[0][0], data.coords[0][1], data.coords[0][2]);
                yield return StartCoroutine(AudioGeneration(data.audio[0], targetPosition));
            }
            else
            {
                Vector3 targetPosition = new Vector3(data.coords[0][0], data.coords[0][1], data.coords[0][2]);
                Vector3 distPosition = new Vector3(data.coords[1][0], data.coords[1][1], data.coords[1][2]);
                yield return StartCoroutine(DoubleAudioGeneration(data.audio[0], data.audio[1], targetPosition, distPosition));
            }
        }
    }

    private IEnumerator AudioGeneration(string filePath, Vector3 position)
    {
        bool clipLoaded = false;

        yield return StartCoroutine(LoadAudioFromFile(filePath, (clip) =>
        {
            audioClip = clip;
            clipLoaded = true;
        }));

        if (audioClip != null)
        {
            PlaceAudioAtLocation(audioClip, position, true);
        }
    }

    private IEnumerator DoubleAudioGeneration(string targetFilePath, string distFilePath, Vector3 targetPos, Vector3 distPos)
    {
        bool targetClipLoaded = false;
        bool distClipLoaded = false;

        yield return StartCoroutine(LoadAudioFromFile(targetFilePath, (clip) =>
        {
            targetAudioClip = clip;
            targetClipLoaded = true;
        }));

        yield return StartCoroutine(LoadAudioFromFile(distFilePath, (clip) =>
        {
            distAudioClip = clip;
            distClipLoaded = true;
        }));

        yield return new WaitUntil(() => targetClipLoaded && distClipLoaded);

        if (targetAudioClip != null && distAudioClip != null)
        {
            PlaceAudioAtLocation(targetAudioClip, targetPos, false);
            PlaceAudioAtLocation(distAudioClip, distPos, true);
        }
    }

    private void PlaceAudioAtLocation(AudioClip clip, Vector3 pos, bool sendData)
    {
        GameObject audioObj = new GameObject("SoundObj");
        audioObj.transform.position = pos;

        AudioSource audio = audioObj.AddComponent<AudioSource>();
        audio.clip = clip;
        audio.spatialBlend = 1.0f;
        audio.spatialize = true;
        audio.volume = 1.0f;
        audio.loop = true;
        audio.rolloffMode = AudioRolloffMode.Logarithmic;
        audio.minDistance = 0.76f;
        audio.maxDistance = 1.5f;

        SteamAudioSource steamAudioSource = audioObj.AddComponent<SteamAudioSource>();
        steamAudioSource.directBinaural = true;
        steamAudioSource.distanceAttenuation = true;

        audio.Play();

        if (sendData)
        {
            writer.WriteLine("done");
            writer.Flush();
        }
        ;
        audioObjects.Add(audioObj);
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
            AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
            Debug.Log($"Successfully loaded external clip of Length: {clip.length} seconds, Channels: {clip.channels}");
            onLoaded?.Invoke(clip);
        }
    }
}