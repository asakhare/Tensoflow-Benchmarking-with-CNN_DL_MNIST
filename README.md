## Bridges Deep Learning benchmarking suite
Deep Learning Benchmarks on Bridges

This repository includes code to create performance benchmarks for the hardware on Bridges which is High Performance Computing platform managed by Pittsburgh Supercomputing Center. This repository includes a benchmarking suite which can be used to execute and record performance benchmarks on Bridges for deep learning training. Currently, the benchmarking suite includes Deep MNIST model which is a classification problem for images of handwritten digits.

There are two directories which include codes to be run on different deployment architectures.
  
#### 1. Deep_MNIST_SingleNode:

This directory includes a benchmarking suite with Deep MNIST model, which can be run on Single Node. The benchmarks are created for various combinations of deployment architecture. Below are some of the combinations of hardware that are benchmarked in Bridges using this suite.

**a. CPU Benchmarks :**
  Record the performance of CPUs in training a Deep Learning Model. These experiments can be run on all types of nodes/machines (including below given example on Bridges) which have atleast one CPU.
  
  i. Regular Memory (RM) Partition on Bridges
  
  ii. Large Memory (LM) Partition on Bridges
  
  iii. Extra Large Memory (XLM) Partition on Bridges
  
  iv. GPU Partitions (CPUs on K80 and P100 Nodes on Bridges).

**b. GPU Benchmarks**
  Record the performance of GPUs in training a Deep Learning Model. These experiments can be run on all types of nodes/machines (including below given example on Bridges) which have atleast one GPU.
  E.g. GPU Partitions (K80 and P100 Nodes on Bridges)
  
**Below is the detailed description of different scripts that gives a high level idea of how the framework carries out the execution of benchmarking experiments.**
<p align="center">
  <img src="Benchmarking_Framework.png" title="Benchmarking_Framework">
</p>

More details for executing the benchmarking suite are in the README file at https://github.com/pscedu/Bridges_DL_benchmarks/tree/master/Deep_MNIST_SingleNode

**2. Deep_MNIST_Distributed_MultiNode**

This directory includes code which implements Deep MNSIT model in a distributed mode. In Distributed Tensorflow, the model employs multiple GPU nodes and distributes the computation across the hardware present on these nodes. Currently as a proof of concept, a code which uses only single GPUs across the nodes is implemented successfully. Going forward we can implement a model to employ Multiple Nodes with Multiple GPUs for training the deep neural network.
(Code modified from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py)
Note - This benchmarking suite is still under development and currently the code to execute Deep MNIST on Multiple Nodes with Single GPUs is working. 
