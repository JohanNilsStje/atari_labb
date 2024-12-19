# Deep Q-Learning with Space Invaders

This repository contains my implementation of the Deep Q-Learning approach to train an AI agent to play the game Space Invaders. The project includes training the model, evaluating its performance, and documenting the process and results.

---

## 1. **Project Overview**

This project implements a Deep Q-Learning (DQN) algorithm to train an agent to play Space Invaders using the Gymnasium library. The training process simulates the game during learning, using the raw game frames as the observation space.

---

## 2. **Features**

- **Deep Q-Learning Implementation**: Built using PyTorch and Gymnasium.
- **Environment Wrappers**: Preprocess game frames (grayscale, resize, and stack).
- **Custom Reward Conditions**: Adjusted rewards for better training efficiency.
- **Model Checkpointing**: Save the model periodically during training.
- **Evaluation**: Test the trained model over multiple runs and visualize results.

---

## 3. **Repository Structure**

├── train.py # Script to train the DQN model 
├── test_model.py # Script to evaluate the trained model 
├── utils.py # Helper functions and classes 
├── config.py # Configuration parameters 
├── requirements.txt # Python dependencies 
├── models/ # Directory for saved models 
├── runs/ # TensorBoard logs 
└── README.md # Project documentation


---

## 4. **Setup Instructions**

### Prerequisites
- Python 3.8 or later
- `pip` for dependency management
- CUDA 11.8

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JohanNilsStje/atari_labb.git
   cd atari_labb

2. **Create a Virtual Environment:**
   ```bash
  python -m venv .venv
  .venv\Scripts\activate

3. **Install Dependencies**
  ```bash
  pip install -r requirements.txt
4. **Train the model**
  ```bash
  python train.py 
5. **Test the model**
  ```bash
  python test_model.py 

### Implementation Details

**Deep Q-Learning**
The Deep Q-Learning algorithm approximates the Q-function using a deep neural network. The model learns to predict the optimal action-value for each state-action pair.

**Model Architecture**
1. 3 Convolutional layers to process game frames
2. Fully connected layers to output Q-values for each action
**Optimizations**
1. Frame preprocessing: Grayscale, resizing to 84x84, and stacking 4 frames
2. Experience replay: Stores transitions and samples batches to stabilize training
3. Target network: Decouples Q-value updates to improve convergence
**Training Metrics**
1. Reward per episode logged to TensorBoard
2. Periodic model saving to models/

### Results

**Training Performance**
1. Episodes Trained: 20000
2. Max Reward Achived:

**Testing Performance**
1. Average Reward:
2. INSERT hur modellen spelade under testrundorna

