MNIST MPI/SGE
=============

Proof-of-concept demo showing how to train a neural network on a cluster of CPU-only servers on an SGE-based
high-performance computing cluster using the OpenMPI parallel environment.

Note that this design isn't optimized for speed; in particular, since MNIST is a very simple dataset that can
be successfully fitted with a small network (e.g. with a 4-layer convolutional/feedforward network), we'll use
this approach to demonstrate how MPI can be used to train models on a cluster of CPU-only servers.


Requirements
------------
* PyTorch v0.4.0+ compiled with MPI support
* an existing installation of an MPI library


Distributed Architecture
------------------------
* We use a master-worker architecture in which we have a central parameter server and a user-specified
  number of workers. Each worker trains a copy of a convolutional neural network model over an epoch through
  the training set, and sends the trained model's state dict back to the master server, which then averages
  the weights together and then runs a validation loop. This is considered to be one distributed epoch.


MPIRUN on SGE
-------------
In general, you can run MPI-based programs on SGE clusters with the following commands:

For Python programs:
```
$ qrsh -l h_vmem=15G -pe mpi <NHOSTS>    # request a shell with multiple hosts
$ mpirun -n <NPROCESSES> python your_spmd_program_here.py
```

For compiled binaries in general (use `mpicc` to compile instead of `gcc`):
```
$ qrsh -l h_vmem=15G -pe mpi <NHOSTS>
$ mpirun -n <NPROCESSES> ./binary_program
```

Training a network with our demo script in particular can be done via the following:
```
$ qrsh -l h_vmem=15G -pe mpi 4
$ mpirun -n 4 python dist_train.py
```
Hyperparameters (learning rate, momentum, batch size, number of epochs) can be set with the appropriate flags;
see `$ python dist_train.py --help` for more details.

On my computer, the validation loss on MNIST is roughly 0.035% after 10 epochs.

License
-------
Aside from the `MNISTNetwork` model itself (in `model.py`), which was inspired by the [official MNIST Hogwild example](https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py#L29)
from the PyTorch authors (but not identical to it), everything in here is my own work and free to be used for whatever purposes you'd like. Consider this to be equivalent to [WTFPL](https://en.wikipedia.org/wiki/WTFPL) aside from the aforementioned model.
