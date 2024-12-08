import os
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
from clean import downsample_mono, envelope
import argparse
import queue
import csv
import datetime
import threading
import sys
import asyncio
import ssl
import logging
import requests

try:
    import geocoder
except ImportError:
    geocoder = None

audio_buffer = queue.Queue()

async def send_message(latitude, longitude, caliber, confidence):
    tak_server = "172.20.10.6"
    tak_port = 8089
    confidence = float(confidence)
    message = f"""<?xml version='1.0' standalone='yes'?>
    <event version="2.0" uid="SoundResponder" type="a-f-G-U-C" how="m-g">
        <point lat="{latitude}" lon="{longitude}" hae="0.0" ce="9999999.0" le="9999999.0"/>
        <detail>
            <contact callsign="SoundResponder"/>
            <remarks>Caliber detected: {caliber} with a {round(confidence, 2)} accuracy</remarks>
        </detail>
    </event>"""

    client_cert = os.path.expanduser("Coms-Cert/wintak_cert.pem")
    client_key = os.path.expanduser("Coms-Cert/wintak_key.pem")
    ca_cert = os.path.expanduser("Coms-Cert/TAK-Sound.pem")

    assert os.path.exists(client_cert), f"Client cert not found: {client_cert}"
    assert os.path.exists(client_key), f"Client key not found: {client_key}"
    assert os.path.exists(ca_cert), f"CA cert not found: {ca_cert}"

    ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_cert)
    ssl_context.load_cert_chain(certfile=client_cert, keyfile=client_key)
    ssl_context.check_hostname = False

    try:
        reader, writer = await asyncio.open_connection(tak_server, tak_port, ssl=ssl_context)
        writer.write(message.encode())
        await writer.drain()
        response = await reader.read(4096)
        if response:
            print("Server response:", response.decode(), flush=True)
        else:
            print("No response from the server.", flush=True)
        print("Message sent successfully.", flush=True)
    except ssl.SSLError as ssl_err:
        logging.exception("SSL error occurred while sending the message.")
    except OSError as os_err:
        logging.exception("OS error occurred, could be network-related.")
    except Exception as e:
        logging.exception("An unexpected error occurred.")
    finally:
        writer.close()
        await writer.wait_closed()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr, flush=True)
    audio_buffer.put(indata.copy())

def preprocess_audio(wav, sr, dt, threshold):
    mask, env = envelope(wav, sr, threshold=threshold)
    clean_wav = wav[mask]
    step = int(sr * dt)
    batch = []

    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i+step]
        sample = sample.reshape(-1, 1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0], :] = sample.flatten().reshape(-1, 1)
            sample = tmp
        batch.append(sample)

    return np.array(batch, dtype=np.float32)

def get_gps_coordinates():
    if geocoder is None:
        print("Geocoder library not installed. GPS data will not be recorded.", flush=True)
        return None, None
    try:
        g = geocoder.ip('me')
        if g.ok:
            latlng = g.latlng
            print(f"Obtained GPS coordinates: Latitude = {latlng[0]}, Longitude = {latlng[1]}", flush=True)
            return latlng
        else:
            print("Could not obtain GPS coordinates.", flush=True)
            return None, None
    except Exception as e:
        print(f"Error obtaining GPS data: {e}", flush=True)
        return None, None

def send_to_remote_database(event_time, latitude, longitude, caliber, confidence):
    print("Debug: Entering send_to_remote_database()", flush=True)

    latitude = float(latitude) if latitude is not None else None
    longitude = float(longitude) if longitude is not None else None
    confidence = float(confidence)

    url = "http://172.20.10.6:5000/insert_event"
    payload = {
        "event_time": event_time,
        "latitude": latitude,
        "longitude": longitude,
        "caliber": caliber,
        "confidence": confidence
    }

    print("Debug: Sending request to remote API...", flush=True)
    try:
        response = requests.post(url, json=payload, timeout=20)
        print(f"Debug: Response received. Status: {response.status_code}", flush=True)
        if response.status_code == 200:
            print("Successfully wrote to remote database via API.", flush=True)
        else:
            print(f"Failed to write to remote database. Status code: {response.status_code}", flush=True)
            print("Response:", response.text, flush=True)
    except requests.RequestException as e:
        print(f"Error connecting to remote database API: {e}", flush=True)

def live_prediction(args):
    model = load_model(args.model_fn, custom_objects={
        'STFT': STFT,
        'Magnitude': Magnitude,
        'ApplyFilterbank': ApplyFilterbank,
        'MagnitudeToDecibel': MagnitudeToDecibel
    })
    classes = sorted(os.listdir(args.src_dir))
    le = LabelEncoder()
    le.fit(classes)

    csv_file = open(args.csv_filename, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        csv_writer.writerow(['Time', 'Latitude', 'Longitude', 'Caliber', 'Confidence'])

    def process_audio():
        while True:
            if not audio_buffer.empty():
                try:
                    audio_chunk = audio_buffer.get()
                    audio_chunk = audio_chunk.flatten()
                    X_batch = preprocess_audio(audio_chunk, args.sr, args.dt, args.threshold)

                    if X_batch.size > 0:
                        y_pred = model.predict(X_batch)
                        y_mean = np.mean(y_pred, axis=0)
                        y_pred_class_index = np.argmax(y_mean)
                        confidence = y_mean[y_pred_class_index] * 100
                        predicted_caliber = classes[y_pred_class_index]

                        print(f"Predicted Caliber: {predicted_caliber}, Confidence: {confidence:.2f}%", flush=True)

                        if confidence >= args.confidence_threshold:
                            event_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            latitude, longitude = get_gps_coordinates()

                            csv_writer.writerow([event_time, latitude, longitude, predicted_caliber, f"{confidence:.2f}%"])
                            csv_file.flush()

                            print("Debug: Calling send_to_remote_database()", flush=True)
                            send_to_remote_database(event_time, latitude, longitude, predicted_caliber, confidence)

                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(send_message(latitude, longitude, predicted_caliber, confidence))
                            loop.close()

                            if latitude is not None and longitude is not None:
                                print(
                                    f"Gunshot detected ({predicted_caliber}) at {event_time} with {confidence:.2f}% confidence.",
                                    flush=True)
                                print(f"Location: Latitude = {latitude}, Longitude = {longitude}", flush=True)
                            else:
                                print(
                                    f"Gunshot detected ({predicted_caliber}) at {event_time} with {confidence:.2f}% confidence.",
                                    flush=True)
                                print("Location: GPS coordinates not available.", flush=True)
                except Exception as e:
                    logging.exception("An error occurred in process_audio.")

    processing_thread = threading.Thread(target=process_audio, daemon=True)
    processing_thread.start()

    if args.test_file:
        from scipy.io import wavfile
        import time

        print(f"Testing with audio file: {args.test_file}", flush=True)
        sr, audio_data = wavfile.read(args.test_file)
        if sr != args.sr:
            print(f"Sample rate mismatch: {sr} Hz vs expected {args.sr} Hz.", flush=True)
        audio_data = audio_data.astype(np.float32)

        step = int(args.sr * args.dt)
        for i in range(0, len(audio_data), step):
            chunk = audio_data[i:i+step]
            if len(chunk) < step:
                padding = np.zeros(step - len(chunk), dtype=np.float32)
                chunk = np.concatenate((chunk, padding))
            chunk = chunk.reshape(-1, 1)
            audio_buffer.put(chunk)
            time.sleep(args.dt)
        time.sleep(2)
    else:
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=args.sr, blocksize=int(args.sr * args.dt), device=args.device):
            print("Listening... Press Ctrl+C to stop.", flush=True)
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("Stopped.", flush=True)
            finally:
                csv_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live Audio Classification with Gunshot Detection')
    parser.add_argument('--model_fn', type=str, default='models/conv1d.keras', help='Model file to use for predictions.')
    parser.add_argument('--src_dir', type=str, default='wavfiles', help='Directory containing class labels.')
    parser.add_argument('--dt', type=float, default=1.0, help='Time in seconds to sample audio.')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate of clean audio.')
    parser.add_argument('--threshold', type=int, default=20, help='Threshold magnitude for np.int16 dtype.')
    parser.add_argument('--confidence_threshold', type=float, default=50.0, help='Confidence threshold (%) to consider a detection.')
    parser.add_argument('--csv_filename', type=str, default='gunshot_events.csv', help='CSV filename to save detected events.')
    parser.add_argument('--device', type=int, default=None, help='Input device index for microphone selection.')
    parser.add_argument('--test_file', type=str, default=None, help='Path to an audio file for testing instead of live microphone input.')
    args = parser.parse_args()

    if args.test_file is None and args.device is None:
        print("Available audio input devices:", flush=True)
        devices = sd.query_devices()
        input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
        for i in input_devices:
            print(f"{i}: {devices[i]['name']}", flush=True)
        device_index = int(input("Select the device index: "))
        args.device = device_index

    live_prediction(args)
