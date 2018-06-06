# Deep-Neural-Networks--Step-by-Step
### Objectives
<p> The outline of the assigment to build a deep neural network for a image classification task is as follows: <p>

* Initialize the parameters for a L layer neural network.
* Implement the forward propagation module to compute the activation functions for L layers. We calculate RELU function for L-1 layers and Sigmoid function for the Lth layer. We store Z as a cache to be used in while calculating gradients in the backward propagation.
* Compute the loss.
* Implement the backward propagation module to compute the gradients of activation function and parameters.
* Update the parameters using gradient descent method.

## Notation
- Superscript [*l*] denotes a quantity associated with the *l^{th}* layer. 
    - Example: *a<sup>[L]<sup>* is the *L<sup>th<sup> layer activation. *W^{[L]}* and *b^{[L]}* are the *L^{th}* layer parameters.
- Superscript $*(i)* denotes a quantity associated with the *i^{th}* example. 
    - Example: *x^{(i)}* is the *i^{th}* training example.
- Lowerscript *i* denotes the *i^{th}* entry of a vector.
    - Example: *a^{[l]}_i* denotes the *i^{th}* entry of the *l^{th}* layer's activations).
    
## Importing Libraries

## Initialization
<p> We write a helper function to initialize the parameters. <p>
