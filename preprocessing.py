import os
import librosa.util
import pandas as pd
import numpy as np
import soundfile as sf  # For saving audio
import librosa
import scipy.signal
import noisereduce as nr
import matplotlib.pyplot as plt

base_path = r"dataset"

target_duration = 10  # seconds
target_sr = 4000  # Hz

spectrogram_npy_path = 'spectrograms_npy'
spectrogram_png_path = 'spectrograms_png'
os.makedirs(spectrogram_npy_path, exist_ok=True)
os.makedirs(spectrogram_png_path, exist_ok=True)

set_a = pd.read_csv(os.path.join(base_path, 'set_a.csv'))
set_b = pd.read_csv(os.path.join(base_path, 'set_b.csv'))
set_a['set'] = 'set_a'
set_b['set'] = 'set_b'
df = pd.concat([set_a, set_b], ignore_index=True)
# df = df.dropna(subset=['label'])

def map_label(label):
    if label == 'murmur':
        return 'murmur'
    elif label == 'normal':
        return 'normal'
    else:
        return np.nan

df['label'] = df['label'].apply(map_label)
df = df.dropna(subset=['label'])
df = df.drop(columns=['sublabel'])

# Clean and join full path safely
def fix_path(fname):
    return os.path.join(base_path, *fname.split("/"))


# this is to fix the naming mismatch in the dataset
def clean_fname(fname):
    if fname.startswith('set_b/'):
        fname = fname.replace('set_b/Btraining_murmur_Btraining_noisymurmur_', 'set_b/murmur_noisymurmur_')
        fname = fname.replace('set_b/Btraining_normal_Btraining_noisymurmur_', 'set_b/normal_noisymurmur_')
        fname = fname.replace('set_b/Btraining_normal', 'set_b/normal_')
        fname = fname.replace('set_b/Btraining_murmur', 'set_b/murmur_')
        fname = fname.replace('set_b/normal__Btraining', 'set_b/normal')
    return fname
df['fname'] = df['fname'].apply(clean_fname)
df['full_path'] = df['fname'].apply(fix_path)
df.to_csv("dataset\\cleaned_csv.csv", index=False)

# Trimming and omiting

output_dirs = {
    'set_a': os.path.join(base_path, 'set_a_trimmed'),
    'set_b': os.path.join(base_path, 'set_b_trimmed')
}
os.makedirs(output_dirs['set_a'], exist_ok=True)
os.makedirs(output_dirs['set_b'], exist_ok=True)

chunk_duration = 3  # seconds
chunk_length = target_sr * chunk_duration

trimmed_data = []

# For trimming and spliting
for idx, row in df.iterrows():
    try:
        y, sr = librosa.load(row['full_path'], sr=target_sr)
    except Exception as e:
        print(f"Could not load {row['full_path']}: {e}")
        continue

    if len(y) < chunk_length:
        continue  # Skip too-short recordings

    label = row['label']
    set_name = row['set']
    filename_base = os.path.splitext(os.path.basename(row['fname']))[0]

    if label == 'normal':
        # Trim first 3 seconds
        y_trimmed = y[:chunk_length]
        out_fname = f"{filename_base}.wav"
        out_path = os.path.join(output_dirs[set_name], out_fname)

        sf.write(out_path, y_trimmed, sr)
        trimmed_data.append({
            'file': out_path,
            'label': label,
            'original_filename': row['fname'],
            'chunk_index': 0
        })

    elif label == 'murmur':
        total_chunks = len(y) // chunk_length
        for i in range(total_chunks):
            start = i * chunk_length
            end = start + chunk_length
            chunk = y[start:end]

            chunk_fname = f"{filename_base}_chunk{i}.wav"
            out_path = os.path.join(output_dirs[set_name], chunk_fname)

            sf.write(out_path, chunk, sr)
            trimmed_data.append({
                'file': out_path,
                'label': label,
                'original_filename': row['fname'],
                'chunk_index': i
            })

# Save trimmed metadata
trimmed_df = pd.DataFrame(trimmed_data)
trimmed_df.to_csv(os.path.join(base_path, 'trimmed_data.csv'), index=False)
print(f"Saved trimmed data metadata to {os.path.join(base_path, 'trimmed_data.csv')}")


# for noise reduction
def reduce_noise_file(file_path):
    data, rate = librosa.load(file_path, sr=None)
    if len(data) < rate:
        time_mask_ms = 64
    else:
        time_mask_ms = 128
    reduced_noise = nr.reduce_noise(y=data, sr=rate, time_mask_smooth_ms=time_mask_ms)
    sf.write(file_path, reduced_noise, rate)


def bandpass_filter(y, sr, lowcut=25.0, highcut=400.0):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    y_filtered = scipy.signal.filtfilt(b, a, y)
    return y_filtered

spectrogram_list = []
png_list = []
label_list = []

for idx, row in trimmed_df.iterrows():
    file_path = row['file']
    label = row['label']
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        reduce_noise_file(file_path)
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        y = bandpass_filter(y, sr)
        y = librosa.util.normalize(y)

        # Mel spectrogram extraction
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=800)
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Save spectrogram as numpy array
        spectrogram_filename = f"{base_name}.npy"
        np.save(os.path.join(spectrogram_npy_path, spectrogram_filename), S_DB)

        # Save spectrogram as image
        plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr, cmap='magma')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(spectrogram_png_path, f"{base_name}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        png_name = f"{base_name}.png"
        # Keep label for metadata
        spectrogram_list.append(spectrogram_filename)
        png_list.append(png_name)
        label_list.append(label)

        print(f"Processed file {idx+1}/{len(trimmed_df)}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

metadata = pd.DataFrame({
    'filename': spectrogram_list,
    'png': png_list,
    'label': label_list
})

metadata.to_csv('spectrogram_metadata.csv', index=False)
print("Spectrogram extraction complete!")
