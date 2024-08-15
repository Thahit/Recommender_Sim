# Notes
The preprocessing notebook is faulty. For example the rewards column is filled with ints and Bools but the notebook wants floats?
The multiprocessing "utils.parallelize_fnc_groups" does not seem to be working.

The requirements are insufficient, they just tell you to use tuff thats newer than xy version, but thats wrong.
- Tensorflow needs to be old -> need to install 1.15, which means we need python 3.7 or older
- h5py=2.10 matplotlib pytorch=2.3.0

install torchdiffeq, pytorch_warmup, evotorch, functorch, seaborn, tqdm