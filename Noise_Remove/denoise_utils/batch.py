from tqdm import tqdm
import soundfile as sf

from .clear_memory import clear_memory


def save_wav_pcm16(out_path, audio, sr):
    sf.write(str(out_path), audio, sr, subtype='PCM_16')


def save_wav_torchaudio(out_path, audio, sr):
    import torchaudio
    torchaudio.save(str(out_path), audio[None], sr)


def batch_denoise(files, output_subdir, denoise_fn, group_name, save_fn=save_wav_pcm16):
    """
    Generic batch denoising loop.

    Args:
        files: list of input audio file paths
        output_subdir: output directory (Path); created if missing
        denoise_fn: callable(audio_path) -> (audio, sr)
        group_name: label shown in tqdm and final stats
        save_fn: callable(out_path, audio, sr); defaults to 16-bit PCM wav
    """
    output_subdir.mkdir(parents=True, exist_ok=True)

    success = skip = fail = 0

    for audio_file in tqdm(files, desc=f"Processing {group_name}"):
        out_file = output_subdir / (audio_file.stem + '.wav')

        if out_file.exists():
            skip += 1
            continue

        try:
            clear_memory()
            audio, sr = denoise_fn(audio_file)
            save_fn(out_file, audio, sr)
            success += 1
            del audio
            clear_memory()
        except Exception as e:
            fail += 1
            print(f"\nFailed: {audio_file.name}: {e}")
            clear_memory()

    print(f"\n{group_name} complete: success={success}, skip={skip}, fail={fail}, total={len(files)}")
