import pyaudio
import wave
import os
import threading

SAMPLE_RATE = 44100  # Sampling rate in Hz
DURATION = 10  # Duration of recording in seconds
CHUNK = 1024  # Size of each audio chunk for buffering
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of audio channels (mono)

OUTPUT_DIR_USB = "GunShotData/Bad-Mic-USB/Handguns"  # Directory for USB mic recordings
OUTPUT_DIR_AUX = "GunShotData/Good-Mic-AUX/Handguns"  # Directory for AUX mic recordings

# Initialize shot count
shot_count = 0


def list_audio_devices(p):
    """List all available audio devices."""
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info['name']} - {info['maxInputChannels']} input channels")


def save_recording(audio_data, output_file):
    """Save recorded audio data to a WAV file."""
    try:
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        print(f"Saved recording to {output_file}")
    except Exception as e:
        print(f"Error saving recording to {output_file}: {e}")


def record_audio(device_index, duration, output_file):
    """Record audio from a specific device and save it to a WAV file."""
    p = pyaudio.PyAudio()
    try:
        print(f"Recording on device {device_index} for {duration} seconds...")
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                        input=True, frames_per_buffer=CHUNK, input_device_index=device_index)

        frames = []

        for _ in range(0, int(SAMPLE_RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recording
        save_recording(b''.join(frames), output_file)

    except Exception as e:
        print(f"Error recording on device {device_index}: {e}")
        p.terminate()


def dual_recording_loop(usb_device_id, aux_device_id):
    """Handle simultaneous recording from two devices."""
    global shot_count

    # Ensure directories exist
    os.makedirs(OUTPUT_DIR_USB, exist_ok=True)
    os.makedirs(OUTPUT_DIR_AUX, exist_ok=True)

    print("Press Enter to save recording and start a new one. Type 'exit' and press Enter to stop.")

    while True:
        shot_count += 1

        # File paths for this shot
        usb_file = f"{OUTPUT_DIR_USB}/caliber_usb_shot{shot_count}.wav"
        aux_file = f"{OUTPUT_DIR_AUX}/caliber_aux_shot{shot_count}.wav"

        # Create threads for simultaneous recording
        usb_thread = threading.Thread(target=record_audio, args=(usb_device_id, DURATION, usb_file))
        aux_thread = threading.Thread(target=record_audio, args=(aux_device_id, DURATION, aux_file))

        # Start recording threads
        usb_thread.start()
        aux_thread.start()

        # Wait for user input to stop
        user_input = input()
        if user_input.lower() == 'exit':
            print("Exiting...")
            usb_thread.join()
            aux_thread.join()
            break
        else:
            print("Saved recordings and ready for the next shot.")

        # Ensure threads have completed before restarting
        usb_thread.join()
        aux_thread.join()

    print("Dual recording completed.")


def main():
    p = pyaudio.PyAudio()

    # List audio devices for user selection
    list_audio_devices(p)

    try:
        usb_device_id = int(input("Enter the device index for USB MIC: "))
        aux_device_id = int(input("Enter the device index for AUX MIC: "))

        # Validate input channels
        usb_info = p.get_device_info_by_index(usb_device_id)
        aux_info = p.get_device_info_by_index(aux_device_id)

        if usb_info['maxInputChannels'] == 0:
            raise ValueError(f"Device {usb_device_id} ({usb_info['name']}) has no input channels.")
        if aux_info['maxInputChannels'] == 0:
            raise ValueError(f"Device {aux_device_id} ({aux_info['name']}) has no input channels.")

        # Start dual recording loop
        dual_recording_loop(usb_device_id, aux_device_id)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.terminate()


if __name__ == "__main__":
    main()
