"""Generate figures and samples from specified network model and checkpoint.

pixel-tree, 2020."""

import argparse
import os
import sys

import pickle
import numpy as np
import PIL.Image

import config
import dnnlib.tflib as tflib
import generate_figures as gf
from training import misc

# Input parameters.

parser = argparse.ArgumentParser(description=None)

parser.add_argument("-n", "--network",
                    type=str,
                    help="Network ID (e.g. 00001-sgan-tf-1gpu).")
parser.add_argument("-s", "--snapshot",
                    type=str,
                    help="Snapshot (e.g. network-snapshot-000001).")
parser.add_argument("-d", "--dimensions",
                    type=int,
                    help="Specify image size (e.g. 256).",
                    default=256)
parser.add_argument("-m", "--multiple",
                    type=int,
                    help="Number of image samples (e.g. --multiple 1000).")

args = parser.parse_args()

# Determine parameters to use.

if args.network:
    network = args.network
    print("\n" + "Network: " + network)

else:
    sys.exit("\n" + "ERROR: Define network ID (e.g. --network 00001-sgan-tf-1gpu)!")

if args.snapshot:
    snapshot = os.path.abspath("results/" + network + "/" + args.snapshot)
    print("Specified snapshot: " + snapshot + "\n")

else:
    snapshot = misc.locate_network_pkl(network, None)  # find latest snapshot.
    print("No snapshot specified. Falling on latest: " + snapshot + "\n")


# Run program.

def main():

    # Initialise TensorFlow.
    tflib.init_tf()

    # Snapshot path.
    url_custom = os.path.abspath(snapshot)

    # Isolate snapshot ID from path tail.
    snapshot_id = os.path.split(snapshot)[1].split("-")[2].split(".")[0]

    # Relevant paths.
    figures_path = os.path.join(config.result_dir + "/examples/" + network + "_" + snapshot_id + "/figures")
    samples_path = os.path.join(config.result_dir + "/examples/" + network + "_" + snapshot_id + "/samples")

    # Create directories if needed.
    os.makedirs(config.result_dir, exist_ok=True)

    if os.path.exists(figures_path) is not True:
        os.makedirs(figures_path, exist_ok=True)

    if os.path.exists(samples_path) is not True:
        os.makedirs(samples_path, exist_ok=True)

    # Load network.
    url = os.path.abspath(snapshot)
    with open(url, 'rb') as f:
        _G, _D, Gs = pickle.load(f)

    # Print network details.
    Gs.print_layers()

    """Generate figures"""

    gf.draw_uncurated_result_figure(os.path.join(figures_path + "/figure01-uncurated.png"), gf.load_Gs(url_custom), cx=0, cy=0, cw=args.dimensions, ch=args.dimensions, rows=3, lods=[0, 1, 2, 2, 3, 3], seed=5)
    gf.draw_style_mixing_figure(os.path.join(figures_path + "/figure02-style-mixing.png"), gf.load_Gs(url_custom), w=args.dimensions, h=args.dimensions, src_seeds=[639, 701, 687, 615, 2268], dst_seeds=[888, 829, 1898, 1733, 1614, 845], style_ranges=[range(0, 4)]*3+[range(4, 8)]*2+[range(8, 13)])
    gf.draw_noise_detail_figure(os.path.join(figures_path + "/figure03-noise-detail.png"), gf.load_Gs(url_custom), w=args.dimensions, h=args.dimensions, num_samples=100, seeds=[1157, 1012])
    gf.draw_noise_components_figure(os.path.join(figures_path + "/figure04-noise-components.png"), gf.load_Gs(url_custom), w=args.dimensions, h=args.dimensions, seeds=[1967, 1555], noise_ranges=[range(0, 18), range(0, 0), range(8, 18), range(0, 8)], flips=[1])
    gf.draw_truncation_trick_figure(os.path.join(figures_path + "/figure05-truncation-trick.png"), gf.load_Gs(url_custom), w=args.dimensions, h=args.dimensions, seeds=[91, 388], psis=[1, 0.7, 0.5, 0, -0.5, -1])

    """Generate samples"""

    # Multiple samples.
    if args.multiple:
        print("\n" + "Mode: generating", args.multiple, "samples...")
        for x in range(0, args.multiple):
            # Pick latent vector.
            rnd = np.random.RandomState(x)
            latents = rnd.randn(1, Gs.input_shape[1])
            # Generate image.
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
            # Save image.
            png_filename = os.path.join(samples_path + "/" + str(x+1) + '.png')
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

    # Single sample (default mode when --multiple not included).
    else:
        print("\n" + "Mode: generating single sample...")
        # Pick latent vector.
        rnd = np.random.RandomState(5)
        latents = rnd.randn(1, Gs.input_shape[1])
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        # Save image.
        png_filename = os.path.join(samples_path + "/example.png")
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


if __name__ == "__main__":
    main()
