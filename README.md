# CycleGAN using Pytorch

## Train
    >> python main.py --mode train --scope [scope name]
 
* You have to set unique [scope name]

## Test
    >> python main.py --mode test --scope [scope name]

* Set [scope name] to test using scoped network

## Tensorboard
    >> tensorboard --logdir log/[scope] --port [(optinal) 4 digit port number]

Then, click http://localhost:[4 digit port number]

* You can change [4 digit port number]
* 4 digit port number = 6006 (default)

## Keypoint

1. Discriminator 
    * Remove Instance Normalization
      
      If there is Instance Normalization, a generator is not trained.
      
    * Leaky ReLU rate changes from 0.2 to 0.01 
