# Lyft Motion Prediction for Autonomous Vehicles
This is how I tortured my low-end 3GB vram gpu with ~100GB of data in order to predict the motion of vehicles, cyclists, and pedestrians. The solution yielded me 64th place and first Kaggle bronze medal.

## Context ([source](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/overview)) 
>In this competition, you’ll apply your data science skills to build motion prediction models for self-driving vehicles. You'll have access to the largest [Prediction Dataset](https://self-driving.lyft.com/level5/prediction/) ever released to train and test your models. Your knowledge of machine learning will then be required to predict how cars, cyclists,and pedestrians move in the AV's environment.

## Solution summary

### Data
1. Modified the rasterizer provided by Lyft in the l5kit package to get few percent of image rendering speed improvement.
2. Reduced number of image channels (it included history positions of agents + RGB) by creating this fading effect that encoded the historic state of agents. Visualized final input image:
![input image](https://i.imgur.com/OK0k1bX.png)
3. Calculated velocity, acceleration, and yaw change rate basing on the agent's history positions. Used it as an additional model's input (concatenated with features extracted from the image). 
4. Created custom masks of the dataset which simulated "chopped-like" data by sampling agents every 0.x seconds from scenes.

### Model
Inspired by [this](https://arxiv.org/pdf/1809.10732.pdf) article, but with NLLLoss. 
- backbone: xresnet34 with Mish activation function instead of ReLU
- head: fully connected layer with output size of 4096, followed by dropout
- input: image of size (5, 250, 200) + agent's state (velocity, acceleration, yaw change rate)
- output: 3 trajectories + probabilities
### Training
First stage: 
- sample size: 8 milion 
- batch size: 16 (could not fit more into memory)
- optimizer: Ranger
- scheduler: flat and cosine anneal
- augmentation: Cutout
- gradient clipping: AutoClip at 15th percentile

Second stage: 
- sample size: 15 milion 
- batch size: 16
- optimizer: Adam
- scheduler: 2 cosine cycles
- augmentation: Cutout
- gradient clipping: AutoClip at 15th percentile


### What did not work
- deeper and larger architectures 
- more layers, activation in the head of the model
- ensembling by combining trajectories using dynamic time warping algorithm
- ensembling by choosing better model based on its performance on similar samples
- focusing on losers (it slightly improved rare trajectories, but made easy samples harder)
- classification approach (like CoverNet)
- regularization by "flooding" technique