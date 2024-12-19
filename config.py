import ale_py
# Environment settings
ENV_NAME = 'ALE/SpaceInvaders-v5'
STACK_SIZE = 4
OBSERVATION_SHAPE = (84, 84)

# Training parameters
REPLAY_BUFFER_CAPACITY = 10000
BATCH_SIZE = 256
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99
TARGET_UPDATE_FREQUENCY = 1000
NUM_EPISODES = 1000 #How many episodes you want it to train before stopping
MAX_STEPS = 1000
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = 'models/dqn_space_invaders_episode_v2_{episode}.pth' #Remember to change this if you train a second model.
PRETRAINED_MODEL_PATH = 'models/dqn_space_invaders_episode_v2_1899.pth' #Change to 'none.pth if you want a new model, also remember to update this and EPSILON_START if you a training a pretrained model
#models/dqn_space_invaders_episode_v2_1899.pth Är den modellen jag tränat lägnst 

# Logging
TENSORBOARD_LOG_DIR = 'runs/space_invaders_dqn'
