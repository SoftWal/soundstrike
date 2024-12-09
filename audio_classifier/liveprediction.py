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

# For GPS data
try:
    import geocoder
except ImportError:
    geocoder = None

# Buffer for audio data
audio_buffer = queue.Queue()

# Asynchronous function to send message to TAK Server
async def send_message(latitude, longitude, caliber, confidence):
    tak_server = "192.168.0.32"
    tak_port = 8089  # TLS-encrypted TCP port
    confidence = float(confidence)
    # Construct message with dynamic coordinates and caliber
    message = f"""<?xml version='1.0' standalone='yes'?>
    <event version="2.0" uid="SoundResponder" type="a-f-G-U-C" how="m-g">
        <point lat="{latitude}" lon="{longitude}" hae="0.0" ce="9999999.0" le="9999999.0"/>
        <detail>
            <contact callsign="SoundResponder"/>
            <remarks>Caliber detected: {caliber} with a {round(confidence, 2)} accuracy</remarks>
        </detail>
    </event>"""

    # Paths to converted PEM files
    client_cert = os.path.expanduser("Coms-Cert/wintak_cert.pem")
    client_key = os.path.expanduser("Coms-Cert/wintak_key.pem")
    ca_cert = os.path.expanduser("Coms-Cert/TAK-Sound.pem")

    # Verify that the certificate files exist
    assert os.path.exists(client_cert), f"Client cert not found: {client_cert}"
    assert os.path.exists(client_key), f"Client key not found: {client_key}"
    assert os.path.exists(ca_cert), f"CA cert not found: {ca_cert}"

    # Create an SSL context
    ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_cert)
    ssl_context.load_cert_chain(certfile=client_cert, keyfile=client_key)
    ssl_context.check_hostname = False  # Disable hostname checking if necessary

    try:
        # Establish a TCP connection with SSL/TLS
        reader, writer = await asyncio.open_connection(
            tak_server, tak_port, ssl=ssl_context
        )

        # Send the message
        writer.write(message.encode())
        await writer.drain()  # Ensure the message is sent

        # Attempt to read response (if any) from the server
        response = await reader.read(4096)
        if response:
            print("Server response:", response.decode())
        else:
            print("No response from the server.")

        print("Message sent successfully.")

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
    """
    Callback to capture audio in real-time.
    """
    if status:
        print(f"Status: {status}", file=sys.stderr)
    audio_buffer.put(indata.copy())

def preprocess_audio(wav, sr, dt, threshold):
    """
    Preprocess audio in real-time chunks.
    """
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
    """
    Get the current GPS coordinates.
    """
    if geocoder is None:
        print("Geocoder library not installed. GPS data will not be recorded.")
        return None, None
    try:
        g = geocoder.ip('me')
        if g.ok:
            latlng = g.latlng
            print(f"Obtained GPS coordinates: Latitude = {latlng[0]}, Longitude = {latlng[1]}")
            return latlng
        else:
            print("Could not obtain GPS coordinates.")
            return None, None
    except Exception as e:
        print(f"Error obtaining GPS data: {e}")
        return None, None

def live_prediction(args):
    """
    Perform live audio prediction.
    """
    # Load model and set up LabelEncoder
    model = load_model(args.model_fn, custom_objects={
        'STFT': STFT,
        'Magnitude': Magnitude,
        'ApplyFilterbank': ApplyFilterbank,
        'MagnitudeToDecibel': MagnitudeToDecibel
    })
    classes = sorted(os.listdir(args.src_dir))
    le = LabelEncoder()
    le.fit(classes)

    # Open CSV file for logging
    csv_file = open(args.csv_filename, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    # Write header only if the file is empty
    if csv_file.tell() == 0:
        csv_writer.writerow(['Time', 'Latitude', 'Longitude', 'Caliber', 'Confidence'])

    def process_audio():
        while True:
            if not audio_buffer.empty():
                try:
                    # Get audio data from the buffer
                    audio_chunk = audio_buffer.get()
                    # Flatten and preprocess
                    audio_chunk = audio_chunk.flatten()
                    X_batch = preprocess_audio(audio_chunk, args.sr, args.dt, args.threshold)

                    # Predict
                    if X_batch.size > 0:
                        y_pred = model.predict(X_batch)
                        y_mean = np.mean(y_pred, axis=0)
                        y_pred_class_index = np.argmax(y_mean)
                        confidence = y_mean[y_pred_class_index] * 100  # Convert to percentage
                        predicted_caliber = classes[y_pred_class_index]

                        # Print predicted caliber and confidence
                        print(f"Predicted Caliber: {predicted_caliber}, Confidence: {confidence:.2f}%")

                        # Check if confidence is above the threshold
                        if confidence >= args.confidence_threshold:
                            # Get current time
                            event_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            # Get GPS coordinates
                            latitude, longitude = get_gps_coordinates()

                            # Log to CSV
                            csv_writer.writerow(
                                [event_time, latitude, longitude, predicted_caliber, f"{confidence:.2f}%"])
                            csv_file.flush()  # Ensure data is written to file

                            # Sending message to TAK Server with caliber
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(send_message(latitude, longitude, predicted_caliber,confidence))
                            loop.close()

                            # Print message
                            if latitude is not None and longitude is not None:
                                print(
                                    f"Gunshot detected ({predicted_caliber}) at {event_time} with {confidence:.2f}% confidence.")
                                print(f"Location: Latitude = {latitude}, Longitude = {longitude}")
                            else:
                                print(
                                    f"Gunshot detected ({predicted_caliber}) at {event_time} with {confidence:.2f}% confidence.")
                                print("Location: GPS coordinates not available.")
                except Exception as e:
                    logging.exception("An error occurred in process_audio.")

    # Start audio processing thread
    processing_thread = threading.Thread(target=process_audio, daemon=True)
    processing_thread.start()

    if args.test_file:
        # Testing with a pre-recorded audio file
        from scipy.io import wavfile
        import time

        print(f"Testing with audio file: {args.test_file}")
        sr, audio_data = wavfile.read(args.test_file)
        if sr != args.sr:
            print(f"Sample rate of the audio file ({sr} Hz) does not match the expected sample rate ({args.sr} Hz).")
            # Optionally resample the audio here
        audio_data = audio_data.astype(np.float32)

        # Simulate real-time audio by feeding chunks into the buffer
        step = int(args.sr * args.dt)
        for i in range(0, len(audio_data), step):
            chunk = audio_data[i:i+step]
            if len(chunk) < step:
                padding = np.zeros(step - len(chunk), dtype=np.float32)
                chunk = np.concatenate((chunk, padding))
            chunk = chunk.reshape(-1, 1)
            audio_buffer.put(chunk)
            time.sleep(args.dt)  # Wait for dt seconds to simulate real-time
        # Allow some time for processing
        time.sleep(2)
    else:
        # Start audio stream
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=args.sr, blocksize=int(args.sr * args.dt), device=args.device):
            print("Listening... Press Ctrl+C to stop.")
            try:
                while True:
                    sd.sleep(1000)  # Sleep to allow other threads to process
            except KeyboardInterrupt:
                print("Stopped.")
            finally:
                csv_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live Audio Classification with Gunshot Detection')
    parser.add_argument('--model_fn', type=str, default='models/conv1d.keras',
                        help='Model file to use for predictions.')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='Directory containing class labels.')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='Time in seconds to sample audio.')
    parser.add_argument('--sr', type=int, default=16000,
                        help='Sample rate of clean audio.')
    parser.add_argument('--threshold', type=int, default=20,
                        help='Threshold magnitude for np.int16 dtype.')
    parser.add_argument('--confidence_threshold', type=float, default=50.0,
                        help='Confidence threshold (%) to consider a detection.')
    parser.add_argument('--csv_filename', type=str, default='gunshot_events.csv',
                        help='CSV filename to save detected events.')
    parser.add_argument('--device', type=int, default=None,
                        help='Input device index for microphone selection.')
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to an audio file for testing instead of live microphone input.')
    args = parser.parse_args()

    # Microphone selection
    if args.test_file is None and args.device is None:
        print("Available audio input devices:")
        devices = sd.query_devices()
        input_devices = [i for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
        for i in input_devices:
            print(f"{i}: {devices[i]['name']}")
        device_index = int(input("Select the device index: "))
        args.device = device_index

    live_prediction(args)