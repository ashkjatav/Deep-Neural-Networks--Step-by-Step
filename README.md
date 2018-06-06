# <pre>Deep Neural Network- Step by Step</pre>

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
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
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
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z= np.dot(W,A)+b
    
    assert(Z.shape==(W.shape[0],A.shape[1]))
    
    cache= (A,W,b)
    
    return Z, cache
```
#### - Linear Activation Forward
We use two activation functions:
-  **Sigmoid**: 
<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma&space;(Z)=&space;\sigma&space;(W&space;A&space;&plus;&space;b)=&space;\frac{1}{1&plus;e^-{(W&space;A&space;&plus;&space;b)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma&space;(Z)=&space;\sigma&space;(W&space;A&space;&plus;&space;b)=&space;\frac{1}{1&plus;e^-{(W&space;A&space;&plus;&space;b)}}" title="\sigma (Z)= \sigma (W A + b)= \frac{1}{1+e^-{(W A + b)}}" /></a>

```python
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache
```
-  **ReLU**:
<a href="https://www.codecogs.com/eqnedit.php?latex=A=&space;RELU(Z)=&space;max(0,Z)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A=&space;RELU(Z)=&space;max(0,Z)" title="A= RELU(Z)= max(0,Z)" /></a>

```python
def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache
```

```python
def linear_activation_forward(A_prev,W,b,activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation=='sigmoid':
        Z, linear_cache= linear_forward(A_prev,W,b)
        A, activation_cache= sigmoid(Z)
        
    elif activation=='relu':
        Z, linear_cache= linear_forward(A_prev,W,b)
        A, activation_cache= relu(Z)
    
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    cache= (linear_cache, activation_cache)
    return A, cache
```

Now, we will create a function which implements the function `linear_activation_forward` created in the last step with `RELU` *L-1* times and with `Sigmoid` one time for the *L<sup>th</sup>* layer.

In the code below, the variable `AL` denotes activation function for the *L<sup>th</sup>* layer.

```python
def L_model_forward(X, parameters):
    caches=[]
    A=X
    L= len(parameters)//2
    
    for l in range(1,L):
        A_prev=A
        A,cache= linear_activation_forward(A_prev, parameters['W'+str(l)],
                                          parameters['b'+str(l)],
                                          activation='relu')
        caches.append(cache)
    AL, cache= linear_activation_forward(A, parameters['W'+str(L)],
                                          parameters['b'+str(L)],
                                          activation='sigmoid')
    caches.append(cache)
    
    assert(AL.shape==(1, X.shape[1]))
    
    return AL, caches
```

Now we calculate the cross-entropy loss, so as to check whether our model is learning or not. Our objective is to minimize the cost by optimizing the parameters `W` and `b` using gradient descent.

<a href="https://www.codecogs.com/eqnedit.php?latex=Cost=&space;J=&space;-\frac{1}{m}\sum&space;(y^{(i)}log(a^{[L](i)})&plus;(1-y^{(i)})log(1-a^{[L](i)}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Cost=&space;J=&space;-\frac{1}{m}\sum&space;(y^{(i)}log(a^{[L](i)})&plus;(1-y^{(i)})log(1-a^{[L](i)}))" title="Cost= J= -\frac{1}{m}\sum (y^{(i)}log(a^{[L](i)})+(1-y^{(i)})log(1-a^{[L](i)}))" /></a>

```python
def compute_cost(AL,Y):
    m= Y.shape[1]
    
    cost= -(1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost= np.squeeze(cost)
    assert(cost.shape==())
    return cost
```
