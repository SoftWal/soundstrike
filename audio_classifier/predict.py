from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm


def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        confidence = 100 * y_mean[y_pred] 
        confidence_r = round(float(confidence), 2)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('Actual class: {}, Predicted class: {}, Confidence: {}% '.format(real_class, classes[y_pred], confidence_r))
        
        results.append(y_mean)

    np.save(os.path.join('logs', args.pred_fn), np.array(results))
    
def predict_folder(model, folder_path, args, classes):
    """
    Predicts the class of each audio file in a specified folder.

    Parameters:
        model: The loaded Keras model.
        folder_path: Path to the folder containing WAV files.
        args: Arguments containing sample rate (sr), threshold, and dt.
        classes: List of class names.

    Returns:
        A DataFrame with file names, predicted classes, and confidence levels.
    """
    # Get all WAV file paths in the folder
    wav_files = sorted(glob(os.path.join(folder_path, '*.wav')))

    if not wav_files:
        print(f"No WAV files found in {folder_path}.")
        return

    results = []
    for wav_fn in tqdm(wav_files, desc="Processing WAV files"):
        # Preprocess each audio file
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr * args.dt)

        batch = []
        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i + step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0], :] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)

        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        predicted_class = np.argmax(y_mean)
        confidence = 100 * y_mean[predicted_class]
        confidence_r = round(float(confidence), 2)

        results.append({
            "file_name": os.path.basename(wav_fn),
            "predicted_class": classes[predicted_class],
            "confidence": f"{confidence_r}%"
        })

    # Convert results to a DataFrame and print summary
    df_results = pd.DataFrame(results)
    print(df_results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/conv2d.keras',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--folder_path', type=str, default='real_prediction',
                        help='Path to a folder containing multiple WAV files to predict.')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=80,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    #make_prediction(args)
    
    # Load model and classes
    model = load_model(args.model_fn,
                       custom_objects={'STFT': STFT,
                                       'Magnitude': Magnitude,
                                       'ApplyFilterbank': ApplyFilterbank,
                                       'MagnitudeToDecibel': MagnitudeToDecibel})
    classes = sorted(os.listdir(args.src_dir))

    # Predict single WAV file
    if args.folder_path:
        predict_folder(model, args.folder_path, args, classes)
    else:
        make_prediction(args)



