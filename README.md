# Julia and Flux: Building Models in Julia for Fashion-MNIST
As a first foray into Julia, I try out Julia for deep learning by building 
some models for Fashion-MNIST. I'm preferring Fashion-MNIST over MNSIT 
because MNIST is pretty overused and an easy problem for simple convolutional 
networks. It also doesn't really represent any modern computer vision problems.

## Fashion-MNIST
Features are 28x28 grayscale images that are associated with a label of 10 
classes:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## LeNet Architecture
[LeNet-5](https://en.wikipedia.org/wiki/LeNet) was proposed in 1998 and 
comprises of two basic parts:

* Convolutional encoder (2 convolutional layers)
* Dense block (3 fully-connected layers)

### Conclusion
Hyperparameters
| Trainable Parameters | 44426 |
| E | Trouser |


Results
| Trainable Parameters | 44426 |
| Training Loss | loss = 0.0639 |
| Training Accuracy | 97.9967 |
| Test Loss | 0.0639 |
| Test Accuracy | 97.9967 |



## AlexNet Architecture
[AlexNet](https://en.wikipedia.org/wiki/AlexNet) was proposed in 2012 by
Krizhevsky, et al. in one of the most influential papers in deep learning 
["ImageNet Classification with Deep Convolutional Neural Networks"](https://www.cs.cmu.edu/~epxing/Class/10715-14f/reading/imagenet.pdf).
AlexNet is similar LeNet in that it uses blocks of convolutions and 
fully-connected layers, however it improves upon the design by adding 
normalization, dropout, and linear layers.

The model can be outlined in the following manner, taken from [Wikipedia](https://en.wikipedia.org/wiki/AlexNet#Network_design).
$$(CNN\to RN\to MP)^{2}\to (CNN^{3}\to MP)\to (FC\to DO)^{2}\to Linear\to softmax$$

| CNN | convolution with ReLU activation |
| RN | local response normalization |
| MP | max-pooling |
| FC | fully-connected layer with ReLU activation |
| DO | dropout |
| Linear | fully connected layer without activation |

However, we are dealing with classifying features of size 28x28x1 into 
10 classes. We scale AlexNet down into the following model:

