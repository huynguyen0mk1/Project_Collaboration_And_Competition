# Project Collaboration and Competition

## Project Details
This project is part of the Udacity Deep Reinforcement Learning nanodegree. We work here with the tennis environment.
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started
To set up your python environment to run the code in this repository, follow the instructions below.

1. **Install required dependencies.** You can install all required dependencies using the provided `setup.py` and `requirements.txt` file.

    ```bash
    !pip -q install .
    ```

2. **Download the Unity Environment.** Select the environment that matches your operating system from one of the links below:
    - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

3. **Unzip the environment file and place it in the root directory of this repository.**

## Instructions
To run the code, follow the Getting started instructions, git clone this repository and go to the folder repository. Then just type:

python Test.ipynb