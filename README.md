# Deep-Neural-Networks--Step-by-Step
### Objectives
<p> The outline of the assigment to build a deep neural network for a image classification task is as follows: <p>

* Initialize the parameters for a L layer neural network.
* Implement the forward propagation module to compute the activation functions for L layers. We calculate RELU function for L-1 layers and Sigmoid function for the Lth layer. We store Z as a cache to be used in while calculating gradients in the backward propagation.
* Compute the loss.
* Implement the backward propagation module to compute the gradients of activation function and parameters.
* Update the parameters using gradient descent method.

## Notation
- Superscript [*l*] denotes a quantity associated with the *l<sup>th</sup> layer. 
    - Example: *a<sup>[L]</sup>* is the *L<sup>th</sup>* layer activation. *W<sup>{[L]}</sup> and *b<sup>{[L]}</sup>* are the *L<sup>{th}</sup>* layer parameters.
- Superscript *(i)* denotes a quantity associated with the *i<sup>{th}</sup>* example. 
    - Example: *x<sup>{(i)}</sup>* is the *i<sup>{th}</sup>* training example.
- Lowerscript *i* denotes the *i<sup>{th}</sup>* entry of a vector.
    - Example: *a<sup>{[l]}_i</sup>* denotes the *i<sup>{th}</sup>* entry of the *l<sup>{th}</sup>* layer's activations).
    
## Importing Libraries

## Initialization
<p> We write a helper function to initialize the parameters. <p>
