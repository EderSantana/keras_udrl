# keras_udrl
Keras implementation of Upside Down Reinforcement Learning

* This is meant to be as small as possible, so it's not as flexible the authors' implementation [here](https://colab.research.google.com/drive/1ynS9g7YzFpNSwhva2_RDKYLjyGckCA8H?usp=sharing#scrollTo=Ypw6MFWIovhC) (in pytorch).
* Behavior function model assumes a gated (multiplicative) function between state and commands in the first few layers. But this should be super easy to change in Keras (have fun messing around).
