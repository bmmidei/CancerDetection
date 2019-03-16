# CancerDetection
This repo is a development space for the Kaggle Competition on cancer detection. The goal is to identify metastatic
cancer in images of digital pathology scans. Our methodology is to use a 3-layer convolutional neural network with
a binary output to classify these images.

More information on the competition can be found [here](https://www.kaggle.com/c/histopathologic-cancer-detection/)

## Results
We were able to achieve an ROC score of 0.82. This score does not place us in the top percentage of competition
entrants. However, as TensorFlow users, our goal was to learn some of the nuances of PyTorch implementations of 
neural networks. To this end, we have succeeded. 

## Reproducing Results

Following these instructions will get you a copy of the project up and running on your local machine for development
and testing purposes.

### Prerequisites

Running the code in this repository requires elementary knowledge of both Jupyter and Anaconda. It is recommended that 
new users create a new virtual environment with Anaconda to ensure that package dependencies match the developer 
versions. If you are unfamiliar with Anaconda, you can find more information and getting started tutorials here:
https://conda.io/docs/user-guide/overview.html

Note that python version 3.6.7 was used for this project. To create a new Anaconda environment, you may use the terminal
command:
```
conda create -n name_of_myenv python=3.6.7
```
After creating this environment, you may clone this repository to your local machine. Within the top level directory,
you will find a 'req.txt' file, which includes a comprehensive list of dependencies necessary to execute the
functionality
of this repository. With your new environment active, use the following command to install these dependencies:
```
pip install -r /path/to/req.txt
```

## Built With
* [PyTorch](https://pytorch.org/) - The Neural Network framework used
* [NumPy](http://www.numpy.org/) - Matrix operations and linear algebra
* [Pandas](https://pandas.pydata.org/) - Data preparation

## Authors

* **Brian Midei** - [bmmidei](https://github.com/bmmidei)
* **Marko Mandic** - [markomandic](https://github.com/markomandic)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
