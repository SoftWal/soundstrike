import os
import numpy as np
import sounddevice as sd
from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
from clean import envelope
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
from scipy.signal import butter, lfilter

try:
    import geocoder
except ImportError:
    geocoder = None

audio_buffer = queue.Queue()

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

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
    except Exception as e:
        logging.exception("Error while sending the message.")
    finally:
        writer.close()
        await writer.wait_closed()

def preprocess_audio(wav, sr, dt, threshold):
    print(f"Preprocessing audio chunk of size: {wav.shape}")
    print(f"Audio chunk min: {np.min(wav)}, max: {np.max(wav)}, mean: {np.mean(wav)}")

    # Apply bandpass filter
    wav = bandpass_filter(wav, lowcut=20, highcut=1000, fs=sr)
    print(f"Bandpass-filtered audio: min={np.min(wav)}, max={np.max(wav)}")

    # Filter out quiet audio segments
    if np.max(wav) < 0.1:
        print("Filtered out quiet audio segment.")
        return np.array([])

    # Envelope-based cleaning
    mask, env = envelope(wav, sr, threshold=threshold)
    print(f"Envelope mask sum: {np.sum(mask)}, Total samples: {len(mask)}")
    clean_wav = wav[mask] if np.any(mask) else wav
    print(f"Cleaned audio chunk size: {clean_wav.shape}")

    # Create batches
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

    print(f"Generated {len(batch)} batches for the model.")
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

    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            print("Successfully wrote to remote database via API.", flush=True)
        else:
            print(f"Failed to write to remote database. Status code: {response.status_code}", flush=True)
    except requests.RequestException as e:
        print(f"Error connecting to remote database API: {e}", flush=True)

def process_audio_chunk(audio_chunk, model, classes, le, csv_writer, args):
    try:
        print(f"Processing audio chunk of size: {audio_chunk.shape}")
        X_batch = preprocess_audio(audio_chunk, args.sr, args.dt, args.threshold)

        if X_batch.size > 0:
            print(f"Model input batch shape: {X_batch.shape}")
            y_pred = model.predict(X_batch)
            print(f"Model output: {y_pred}")
            y_mean = np.mean(y_pred, axis=0)
            y_pred_class_index = np.argmax(y_mean)
            confidence = y_mean[y_pred_class_index] * 100
            predicted_caliber = classes[y_pred_class_index]
            print(f"Predicted caliber: {predicted_caliber}, Confidence: {confidence:.2f}%")

            if confidence >= args.confidence_threshold:
                event_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #latitude, longitude = get_gps_coordinates()
                latitude, longitude = 18.373358,-67.193583

                print(f"Gunshot detected: {predicted_caliber} with {confidence:.2f}% confidence!", flush=True)
                # Send to database
                payload = {
                    "event_time": event_time,
                    "latitude": latitude,
                    "longitude": longitude,
                    "caliber": predicted_caliber,
                    "confidence": confidence
                }
                print(f"Sending data to database: {payload}", flush=True)
                send_to_remote_database(event_time, latitude, longitude, predicted_caliber, confidence)

                # Send message to TAK server
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print(f"Sending TAK server message for gunshot detection.", flush=True)
                loop.run_until_complete(send_message(latitude, longitude, predicted_caliber, confidence))
                loop.close()

                # Save event to CSV
                csv_writer.writerow([event_time, latitude, longitude, predicted_caliber, f"{confidence:.2f}%"])
                csv_writer.flush()




            else:
                print(f"Prediction below threshold: {confidence:.2f}%", flush=True)
    except Exception:
        logging.exception("An error occurred in process_audio_chunk.")


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", file=sys.stderr, flush=True)
    print(f"Captured audio chunk shape: {indata.shape}, dtype: {indata.dtype}", flush=True)
    if indata.ndim > 1:
        indata = np.mean(indata, axis=1)  # Convert stereo to mono
    audio_buffer.put(indata.copy())

def process_audio_stream(model, classes, le, csv_writer, args):
    while True:
        if not audio_buffer.empty():
            audio_chunk = audio_buffer.get()
            process_audio_chunk(audio_chunk, model, classes, le, csv_writer, args)

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

    processing_thread = threading.Thread(target=process_audio_stream, args=(model, classes, le, csv_writer, args), daemon=True)
    processing_thread.start()

    if args.test_file:
        from scipy.io import wavfile
        import time

        sr, audio_data = wavfile.read(args.test_file)
        audio_data = audio_data.astype(np.float32)

        step = int(args.sr * args.dt)
        for i in range(0, len(audio_data), step):
            chunk = audio_data[i:i+step]
            if len(chunk) < step:
                padding = np.zeros(step - len(chunk), dtype=np.float32)
                chunk = np.concatenate((chunk, padding))
            chunk = chunk.reshape(-1, 1)
            process_audio_chunk(chunk, model, classes, le, csv_writer, args)
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
    parser.add_argument('--threshold', type=int, default=5, help='Threshold magnitude for np.int16 dtype.')
    parser.add_argument('--confidence_threshold', type=float, default=50.0, help='Confidence threshold (%) to consider a detection.')
    parser.add_argument('--csv_filename', type=str, default='gunshot_events.csv', help='CSV filename to save detected events.')
    parser.add_argument('--device', type=int, default=None, help='Input device index for microphone selection.')
    parser.add_argument('--test_file', type=str, default=None, help='Path to an audio file for testing instead of live microphone input.')
    args = parser.parse_args()

    if args.test_file is None and args.device is None:
        devices = sd.query_devices()
        input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
        for i in input_devices:
            print(f"{i}: {devices[i]['name']}", flush=True)
        device_index = int(input("Select the device index: "))
        args.device = device_index

    live_prediction(args)
