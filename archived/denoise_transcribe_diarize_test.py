import torch
import sys

def health_check():
    print("🔍 Starting Unified ASR Health Check...")
    results = {}

    # Check 1: CUDA & Hardware
    results['CUDA Available'] = torch.cuda.is_available()
    results['GPU Name'] = torch.cuda.get_device_name(0) if results['CUDA Available'] else "N/A"

    # Check 2: DeepFilterNet (The recent hurdle)
    try:
        from df.enhance import init_df
        model, df_state, _ = init_df()
        results['DeepFilterNet'] = "✅ Healthy (Engine Initialized)"
    except Exception as e:
        results['DeepFilterNet'] = f"❌ FAILED: {str(e)}"

    # Check 3: WhisperX
    try:
        import whisperx
        results['WhisperX'] = "✅ Healthy"
    except Exception as e:
        results['WhisperX'] = f"❌ FAILED: {str(e)}"

    # Check 4: Pyannote
    try:
        import pyannote.audio
        results['Pyannote'] = "✅ Healthy"
    except Exception as e:
        results['Pyannote'] = f"❌ FAILED: {str(e)}"

    print("\n--- FINAL REPORT ---")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    health_check()