The main body of the code is from https://github.com/MadryLab/mnist_challenge.

## Format and Rules

The objective of the challenge is to find black-box (transfer) attacks that are effective against our MNIST model.
Attacks are allowed to perturb each pixel of the input image by at most `epsilon=0.3`.
To ensure that the attacks are indeed black-box, we release our training code and model architecture, but keep the actual network weights secret. 

We invite any interested researchers to submit attacks against our model.
The most successful attacks will be listed in the leaderboard above.
As a reference point, we have seeded the leaderboard with the results of some standard attacks.

### The MNIST Model

We used the code published in this repository to produce an adversarially robust model for MNIST classification. The model is a convolutional neural network consisting of two convolutional layers (each followed by max-pooling) and a fully connected layer. This architecture is derived from the [MNIST tensorflow tutorial](https://www.tensorflow.org/get_started/mnist/pros).
The network was trained against an iterative adversary that is allowed to perturb each pixel by at most `epsilon=0.3`.

The random seed used for training and the trained network weights will be kept secret.

The `sha256()` digest of our model file is:
```
14eea09c72092db5c2eb5e34cd105974f42569281d2f34826316e356d057f96d
```
We will release the corresponding model file on October 15th 2017, which is roughly two months after the start of this competition.

### The Attack Model

We are interested in adversarial inputs that are derived from the MNIST test set.
Each pixel can be perturbed by at most `epsilon=0.3` from its initial value.
All pixels can be perturbed independently, so this is an l_infinity attack.

### Submitting an Attack

Each attack should consist of a perturbed version of the MNIST test set.
Each perturbed image in this test set should follow the above attack model. 

The adversarial test set should be formated as a numpy array with one row per example and each row containing a flattened
array of 28x28 pixels.
Hence the overall dimensions are 10,000 rows and 784 columns.
Each pixel must be in the [0,1] range.
See the script `pgd_attack.py` for an attack that generates an adversarial test set in this format.

In order to submit your attack, save the matrix containing your adversarial examples with `numpy.save` and email the resulting file to mnist.challenge@gmail.com. 
We will then run the `run_attack.py` script on your file to verify that the attack is valid and to evaluate the accuracy of our secret model on your examples.
After that, we will reply with the predictions of our model on each of your examples and the overall accuracy of our model on your evaluation set.

If the attack is valid and outperforms all current attacks in the leaderboard, it will appear at the top of the leaderboard.
Novel types of attacks might be included in the leaderboard even if they do not perform best.

We strongly encourage you to disclose your attack method.
We would be happy to add a link to your code in our leaderboard.

## Overview of the Code
The code consists of six Python scripts and the file `config.json` that contains various parameter settings.

### Running the code
- `python train.py`: trains the network, storing checkpoints along
      the way.
- `python eval.py`: an infinite evaluation loop, processing each new
      checkpoint as it is created while logging summaries. It is intended
      to be run in parallel with the `train.py` script.
- `python pgd_attack.py`:  applies the attack to the MNIST eval set and
      stores the resulting adversarial eval set in a `.npy` file. This file is
      in a valid attack format for our challenge.
- `python run_attack.py`: evaluates the model on the examples in
      the `.npy` file specified in config, while ensuring that the adversarial examples 
      are indeed a valid attack. The script also saves the network predictions in `pred.npy`.
- `python fetch_model.py name`: downloads the pre-trained model with the
      specified name (at the moment `adv_trained` or `natural`), prints the sha256
      hash, and places it in the models directory.

### Parameters in `config.json`

Model configuration:
- `model_dir`: contains the path to the directory of the currently 
      trained/evaluated model.

Training configuration:
- `random_seed`: the seed for the RNG used to initialize the network
      weights.
- `max_num_training_steps`: the number of training steps.
- `num_output_steps`: the number of training steps between printing
      progress in standard output.
- `num_summary_steps`: the number of training steps between storing
      tensorboard summaries.
- `num_checkpoint_steps`: the number of training steps between storing
      model checkpoints.
- `training_batch_size`: the size of the training batch.

Evaluation configuration:
- `num_eval_examples`: the number of MNIST examples to evaluate the
      model on.
- `eval_batch_size`: the size of the evaluation batches.
- `eval_on_cpu`: forces the `eval.py` script to run on the CPU so it does not compete with `train.py` for GPU resources.

Adversarial examples configuration:
- `epsilon`: the maximum allowed perturbation per pixel.
- `k`: the number of PGD iterations used by the adversary.
- `a`: the size of the PGD adversary steps.
- `random_start`: specifies whether the adversary will start iterating
      from the natural example or a random perturbation of it.
- `loss_func`: the loss function used to run pgd on. `xent` corresponds to the
      standard cross-entropy loss, `cw` corresponds to the loss function 
      of [Carlini and Wagner](https://arxiv.org/abs/1608.04644).
- `store_adv_path`: the file in which adversarial examples are stored.
      Relevant for the `pgd_attack.py` and `run_attack.py` scripts.

## Example usage
After cloning the repository you can either train a new network or evaluate/attack one of our pre-trained networks.
#### Training a new network
* Start training by running:
```
python train.py
```
* (Optional) Evaluation summaries can be logged by simultaneously
  running:
```
python eval.py
```
#### Download a pre-trained network
* For an adversarially trained network, run
```
python fetch_model.py adv_trained
```
and use the `config.json` file to set `"model_dir": "models/adv_trained"`.
* For a naturally trained network, run
```
python fetch_model.py natural
```
and use the `config.json` file to set `"model_dir": "models/natural"`.
#### Test the network
* Create an attack file by running
```
python pgd_attack.py
```
* Evaluate the network with
```
python run_attack.py
```
