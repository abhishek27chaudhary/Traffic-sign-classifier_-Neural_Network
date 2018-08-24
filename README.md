# Traffic-sign-classifier
### Overview
In this project, I've built a model to classify the traffic signs.The model is trained using convolutional network based on the [LeNet](http://yann.lecun.com/exdb/lenet/) architecture by Yann LeCun. I've used using scikit-learn's pipeline framewok along with various combination of transformer and estimators.

Augmentation and Normalisation of data is also done before feeding in neural network. I've experimented and built different [networks](https://github.com/Jargon4072/Traffic-sign-classifier/tree/master/networks) with different learning rate,epochs and preprocessors. The nework with highest accuracy is used to train the final model.
### Dependencies
This project requires **Python 3.5** and the following Python libraries installed:
- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [Scipy](https://www.scipy.org/)
- [Scitkit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/)
- [OpenCV](http://opencv.org) useful for image processing.
- [This](https://www.pyimagesearch.com/2015/07/20/install-opencv-3-0-and-python-3-4-on-ubuntu/) tutorial is very useful to install opencv. You can also install this via conda `conda install -c https://conda.anaconda.org/menpo opencv3`.
- OpenCV can also be install with apt-get in Ubuntu.

### Dataset
- Dataset used is German Traffic Sign Recognition Bentchmark(GTSRB). [Download the dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)  from here.
- The Training dataset contains 39,209 training images in 43 classes. The test dataset contains 12,630 test images.

### Model Architecture
The model is based on LeNet by Yann LeCun. It is a convolutional neural network designed to recognize visual patterns directly from pixel images with minimal preprocessing. It can also handle hand-written characters very well.
![LeNet](images/lenet_architecture-768x226.jpeg "LeNet Architecture")

More about LeNet can be found in [this](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) paper published by it's author Yann LeCunn and Pierre Sermanet.

Our model is tweeked version of LeNet in which various modifications are done to improve the accuracy of model.
The Detail about our model is in [ Jupyter notebok file](https://github.com/Jargon4072/Traffic-sign-classifier/blob/master/Traffic-sign-classifier.ipynb) of this repository. Open the notebook file by running this command:

```
$jupyter notebok Traffic-sign-classifier.ipynb
```

### Installation
As explained earlier the project uses Python 3.5 and several dependencies. Install correaponding dependencies and clone this repository by firing up terminal and running following command:

```
$git clone https://github.com/Jargon4072/Traffic-sign-classifier.git
```

Now everything is good to go.

### Usage
train.py is used for training the network and saving the checkpoint file generated in checkpoint folder. Configure corresponding paths in train.py and run it by follwing command:

```
$python3 train.py
```

The paths to be cofigured are:
- pipeline folder path. Make sure to downloade all files from [this](https://github.com/Jargon4072/Traffic-sign-classifier/tree/master/pipeline) link and place it in a folder pipeline.
- Dataset path.

After training checkpoints will be saved in checkpoint folder. Load checkpoint and test the model by using test.py from following command:

```
$python3 test.py
```

Make sure to provide above described paths and the path of folder containing images in the test.py.

The accuracy of model can also be checked by using test_accuracy.py file as follows:

```
$python3 test_accuracy.py
```

Again don't forget to edit corresponding path in this one too.
### References and Resources
- Traffic Sign Recognition with Multi-Scale Convolutional Networks|Pierre Sermanet and Yann LeCun: [http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf]
- LeNet Demo (Yann LeCun): [http://yann.lecun.com/exdb/lenet/]
- Gradient-Based Learning Applied to Document Recognition|Pierre Sermanet and Yann LeCun: [http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf]
- Udacity: Self-Driving Car Engineer: Traffic Sign Classifier Project: [https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project]
- Tensorflow Tutorials: [https://www.tensorflow.org/tutorials/]
- Keras with Scikit-Learn Pipeline: [http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/]
- Pipeline implimentation of LeNet: [https://github.com/naokishibuya/car-traffic-sign-classification/tree/master/pipeline]
- Neural Network and Deep learning complete study [http://neuralnetworksanddeeplearning.com]
