# <pre>Deep Neural Network- Step by Step<pre>

### Objectives
<p> The outline of the assigment to build a deep neural network for a image classification task is as follows: <p>

* Initialize the parameters for a L layer neural network.
* Implement the forward propagation module to compute the activation functions for L layers. We calculate RELU function for L-1 layers and Sigmoid function for the Lth layer. We store Z as a cache to be used in while calculating gradients in the backward propagation.
* Compute the loss.
* Implement the backward propagation module to compute the gradients of activation function and parameters.
* Update the parameters using gradient descent method.

### Notation
- Superscript [*l*] denotes a quantity associated with the *l<sup>th</sup>* layer. 
    - Example: *a<sup>[L]</sup>* is the *L<sup>th</sup>* layer activation. *W<sup>[L]</sup>* and *b<sup>[L]</sup>* are the *L<sup>th</sup>* layer parameters.
- Superscript *(i)* denotes a quantity associated with the *i<sup>th</sup>* example. 
    - Example: *x<sup>(i)</sup>* is the *i<sup>th</sup>* training example.
- Lowerscript *i* denotes the *i<sup>th</sup>* entry of a vector.
    - Example: *a<sup>[l]_i</sup>* denotes the *i<sup>th</sup>* entry of the *l<sup>th</sup>* layer's activations.
    
### Importing Libraries
```python
import numpy as np
import h5py
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)
```

### Initialization
We write a helper function to initialize the parameters. 
We store *n<sup>l</sup>*, the number of units in different layers in a variable `layer_dims`. For example, `layer_dims`= [2,4,1] is a neural network with 2 inputs, one hidden layer with 4 neurons, one output layer with one output unit.

```python
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters={}
    L= len(layer_dims)
    
    for l in range(1,L):
        parameters['W'+str(l)]= np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b'+str(l)]= np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters
```
###  Forward propagation module

#### -Linear Forward 
After initializing the parameters, we will now do the forward propagation module. We will implement some basic functions that we will use later when implementing the model. We will complete three functions in this order:

- LINEAR
- LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid. 
- [LINEAR -> RELU] **X** (L-1) -> LINEAR -> SIGMOID (whole model)

The linear forward module (vectorized over all the examples) computes the following equations:

*Z<sup>[l]</sup> = W<sup>[l]</sup>A<sup>[l-1]</sup> +b<sup>[l]</sup>*

where *A<sup>[0]</sup> = X*. 

```python
def linear_forward(A,W,b):
    Z= np.dot(W,A)+b
    
    assert(Z.shape==(W.shape[0],A.shape[1]))
    
    cache= (A,W,b)
    
    return Z, cache
```
#### - Linear Activation Forward
We use two activation functions:
-  **Sigmoid**: *\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}*
