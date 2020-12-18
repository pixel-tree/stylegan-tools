# stylegan-tools

Additional functionality for the NVlabs/stylegan repository, TensorFlow 1.x implementation.

*This is an unofficial and unendorsed extension to the original (link to [license](https://github.com/NVlabs/stylegan/blob/master/LICENSE.txt)).*

**Instructions**

Clone [original repository](https://github.com/NVlabs/stylegan) and follow instructions to install dependencies, train custom models / download pre-trained networks, etc.

Merge scripts from this repository into stylegan-master root. No additional dependencies are required as of initial commit; but as functionality is extended, requirements, if any, will be provided in repo.

**Generating samples**

Use `samples.py` to generate figures and samples from specified network model and checkpoint.

Required parameters:

--network, -n\
Define network ID, e.g., --network 00001-sgan-tf-1gpu.

Optional:

--dimensions, -d\
Specify image size, e.g., --dimensions 1024 (use same as for training -- defaults to 256).

--snapshot, -s\
Snapshot name, e.g., --snapshot network-snapshot-000001.pkl (defaults to finding the latest snapshot if not specified).

--multiple, -m\
Number of image samples to be generated, e.g., --multiple 1000 (defaults to single sample if not specified).

E.g.
```
python samples.py -n 00001-sgan-tf-1gpu -s network-snapshot-012345.pkl -d 256 -m 1000
```

**Development**

Currently considering linear interpolation, video, GIFs, and some other nifty creative tools...
