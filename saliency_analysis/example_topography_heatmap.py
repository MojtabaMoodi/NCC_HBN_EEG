#!/usr/bin/env python3
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# --- Constants ---
STANDARD_CHANNEL_NAMES = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'F1', 'F2', 'F5', 'F6', 'F9', 'F10',
    'AF3', 'AF4', 'AF7', 'AF8', 'AFZ', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
    'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'P7', 'P3', 'PZ', 'P4', 'P8', 'P1', 'P2',
    'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'OZ', 'O1', 'O2', 'C3', 'C4',
    'C1', 'C2', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPZ'
]


def plot_mne_evoked_topo(data, channel_names, title, save_path=None):
    if data.ndim == 2:
        data = np.mean(data, axis=0)

    # Correct naming for MNE
    renamed_channels = [name.replace('FP', 'Fp').replace('Z', 'z') for name in channel_names]

    info = mne.create_info(ch_names=renamed_channels, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='warn')

    evoked = mne.EvokedArray(data.reshape(-1, 1), info, tmin=0)

    # Final tweak for "Exact Scalp" look:
    fig = evoked.plot_topomap(
        times=[0],
        ch_type='eeg',
        cmap='Spectral_r',
        res=128,
        outlines='head',
        contours=4,
        time_unit='s',
        colorbar=True,
        show=False,
        extrapolate='head',    # <--- Adds the "Exact Scalp" constraint
        sphere=(0, 0, 0, 0.09) # <--- Standard head radius in meters
    )

    plt.suptitle(title, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300) # dpi=300 for high-quality publication output
        print(f"✅ Topography saved to: {save_path}")

    plt.show()


def example_1_from_saved_results():
    print("=" * 80)
    print("Example 1: MNE Evoked Plot from Saved Results")
    print("=" * 80)

    results_path = "saliency_results/gender_baseline_4s/saliency_results_vanilla_gradients.npz"
    try:
        data = np.load(results_path)
        channel_importance = data['channel_importance']

        plot_mne_evoked_topo(
            channel_importance,
            STANDARD_CHANNEL_NAMES,
            title="Importance: Gender Classification",
            save_path="mne_evoked_saved.png"
        )
    except FileNotFoundError:
        print(f"⚠️  Results file not found: {results_path}")
        print("   Run saliency analysis first, or adjust the path to your results file.")
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_custom_data():
    print("\n" + "=" * 80)
    print("Example 2: MNE Evoked Plot from Custom Data")
    print("=" * 80)

    np.random.seed(42)
    num_channels = len(STANDARD_CHANNEL_NAMES)
    channel_importance = np.random.rand(num_channels)

    # Boost frontal and central indices
    frontal_indices = [0, 1, 2, 3, 4]
    channel_importance[frontal_indices] += 0.5

    plot_mne_evoked_topo(
        channel_importance,
        STANDARD_CHANNEL_NAMES,
        title="Custom Importance Topography",
        save_path="mne_evoked_custom.png"
    )


def main():
    print("1. From saved saliency results")
    print("2. From custom channel importance data")
    choice = input("\nEnter choice (1-2): ").strip()

    if choice == '1':
        example_1_from_saved_results()
    else:
        example_2_custom_data()


if __name__ == '__main__':
    main()
