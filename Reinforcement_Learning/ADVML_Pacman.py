#!/usr/bin/env python
# coding: utf-8

# # iLykei Lecture Series
# 
# # University of Chicago
# 
# # Advanced Machine Learning and Artificial Intelligence
# 
# # Reinforcement Learning
# 
# ## Notebook 6: Learning Ms. Pac-Man with DQN
# 
# ## Yuri Balasanov, Mihail Tselishchev, &copy; iLykei 2018-2020
# 
# ## HomeWork Done By : Surendiran Rangaraj
# 
# ##### Main text: Hands-On Machine Learning with Scikit-Learn and TensorFlow, Aurelien Geron, &copy; Aurelien Geron 2017, O'Reilly Media, Inc
# 

# In[14]:


rcc=True
colab=False
if colab:
    from google.colab import drive
    drive.mount('/content/drive')


# In[15]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np
import random
import time
import os
import gc

from tensorflow.keras.models import Sequential, clone_model,load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, InputLayer
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

#import 

plt.rcParams['figure.figsize'] = (9, 9)


# ###### Set your working directory to a folder in your Google Drive. This way, if your notebook times out,
# ###### your files will be saved in your Google Drive!

# In[47]:



import os
# the base Google Drive directory
root_dir = "/content/drive/My Drive/"
if rcc:
    root_dir = "/cloud/msca-gcp/surendiran/"
    # choose where you want your project files to be saved
    project_folder = "pacmanRL/code/"
    data_folder = "pacmanRL/data/"
else:
    if colab:
        root_dir = "/content/drive/My Drive/"
        # choose where you want your project files to be saved
        project_folder = "Colab Notebooks/ADVML/code/"
        data_folder = "Colab Notebooks/ADVML/data/"
    else:
        root_dir = "C:\\Users\\rsure\\Desktop\\Uchicago\\AdvMachineLearning\\"
        # choose where you want your project files to be saved
        project_folder = "Reinforcement Learning\\Pacman_Assignment02\\"
        data_folder = "Reinforcement Learning\\Pacman_Assignment02\\data\\"





# In[26]:


# Set your working directory to a folder in your Google Drive. This way, if your notebook times out,
# your files will be saved in your Google Drive!
def set_WD(project_folder):
    # change the OS to use your project folder as the working directory
  os.chdir(root_dir + project_folder)
  print('\nYour working directory was changed to ' + root_dir + project_folder +         "\nYou can also run !pwd to confirm the current working directory." )


def create_directory(project_folder):
  # check if your project folder exists. if not, it will be created.
  if os.path.isdir(root_dir + project_folder) == False:
    os.mkdir(root_dir + project_folder)
    print(root_dir + project_folder + ' did not exist but was created.')


# In[27]:


create_directory(project_folder)
create_directory(data_folder)
set_WD(project_folder)


# # Deep Q-Learning of MS. Pac-Man with Keras
# 
# This notebook shows how to implement a deep neural network approach to train an agent to play Ms.Pac-Man Atari game.
# 
# 
# ## Explore the game
# 
# Use [Gym](https://gym.openai.com/) toolkit that provides both game environment and also a convenient renderer of the game.
# 
# Create an environment.

# In[ ]:


#!python -m atari_py.import_roms "https://drive.google.com/drive/folders/1aZWOt3e5WcnywBtBLoD4r_VuG1r0zCLw?usp=sharing"
#env = gym.make("MsPacman-ram-v0")
#env.action_space  # actions are integers from 0 to 8


# # Deep Q-Learning of MS. Pac-Man with Keras
# 
# Use Custom Module and states

# In[48]:


import json
from mini_pacman import PacmanGame

#path = "/content/drive/My Drive/Colab Notebooks/ADVML/data/"
filenamepath = root_dir+project_folder+"test_params.json"
with open(filenamepath, 'r') as file:
    read_params = json.load(file)
game_params = read_params['params']
env = PacmanGame(**game_params)


# In[49]:


game_params


# In[ ]:


#env = PacmanGame(field_shape=(10,10), nmonsters=2,
#                 ndiamonds=3, nwalls=4, monster_vision_range=1)


# Try to play the game using random strategy:

# In[50]:


obs=env.reset()
done = False
score = 0
rewards=[]
game_scores=[]
for one_game in range(100):
    obs = env.reset()
    while not obs['end_game']:    
        action = random.choice(obs['possible_actions'])
        obs = env.make_action(action)
    game_scores.append(obs['total_score'])
print('Mean: ',np.mean(game_scores),'\nMedian: ',np.median(game_scores))
    #env.render()
    #time.sleep(0.01)
   
env.close()


# ### Observation
# 
# In this environment, observation (i.e. current state) is the RAM of the Atari machine, namely a vector of 128 bytes:

# In[51]:


obs = env.reset()


# Look at that vector:

# In[52]:


print(obs)
print(env.get_obs())


# Player can control agent using actions from 1 to 9 

# In[53]:


from tabulate import tabulate
print(tabulate([[1,'Down-Left'],                 [2,'Down'],                 [3,'Down-Right'],                 [4,'Left'],                 [5,'No Move'],                 [6,'Right'],                 [7,'Up-Left'],                 [8,'Up'],                 [9,'Up-Right']],                headers = ['Action Code','Move'],               tablefmt='orgtbl'))


# Represent current state as a vector of features.

# In[54]:


def get_state(obs):
    v = []
    x,y = obs['player']
    v.append(x)
    v.append(y)
    for x, y in obs['monsters']:
        v.append(x)
        v.append(y)
    for x, y in obs['diamonds']:
        v.append(x)
        v.append(y)
    for x, y in obs['walls']:
        v.append(x)
        v.append(y)
    return v


# In[55]:


print(env.get_obs())


# In[56]:


state = get_state(env.get_obs())
print(state)


# Create a deep neural network that takes byte vector as an input and produces Q-values for state-action pairs.

# ## Creating a DQN-model using Keras
# 
# The following model is of the same general type applied to the cartPole problem.
# 
# Use vanilla multi-layer dense network with relu activations which computes Q-values $Q(s,a)$ for all states $s$ and actions $a$ (with some discount factor $\gamma$).
# This neural network denoted by $Q(s\ |\ \theta)$ takes current state as an input and produces a vector of q-values for all 9 possible actions. Vector $\theta$ corresponds to all trainable parameters.

# In[57]:


def create_dqn_model(input_shape, nb_actions, dense_layers, dense_units):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    #for i in range(dense_layers):
    #    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


# Create a network using specific input shape and action space size. We call this network *online*.

# In[58]:


input_shape = (len(state),)
nb_actions = 9 #obs['possible_actions']
dense_layers = 6
dense_units = 512
print('input_shape: ',input_shape)
print('nb_actions: ',nb_actions)


# In[59]:


online_network = create_dqn_model(input_shape, nb_actions, dense_layers, dense_units)
online_network.summary()
#target_network = clone_model(online_network)
#target_network.set_weights(online_network.get_weights())


# In[61]:


from tensorflow.keras.utils import plot_model
#plot_model(online_network, to_file='online_DenseNetwork.png',show_shapes=True,show_layer_names=True)
plot_model(online_network,show_shapes=True,show_layer_names=True)


# **Score based on Random and Naive Strategy**

# In[62]:


from mini_pacman import test, random_strategy, naive_strategy
test(strategy=random_strategy, log_file='test_pacman_random_log.json')


# In[63]:


test(strategy=naive_strategy, log_file='test_pacman_naive_log.json')


# This network is used to explore states and rewards of Markov decision process according to an $\varepsilon$-greedy exploration strategy:

# In[64]:


def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.choice(n_outputs)  # random action
    else:
        return np.argmax(q_values) + 1         # q-optimal action  ( add 1 to adjust the index(0 to 8) to action(1 to 9))


# Online network stores explored information in a *replay memory*, a double-ended queue (deque).

# In[65]:


from collections import deque

replay_memory_maxlen = 1000000
replay_memory = deque([], maxlen=replay_memory_maxlen)


# So, online network explores the game using $\varepsilon$-greedy strategy and saves experienced transitions in replay memory. 
# 
# In order to produce Q-values for $\varepsilon$-greedy strategy, following the proposal of the [original paper by Google DeepMind](https://www.nature.com/articles/nature14236), use another network, called *target network*, to calculate "ground-truth" target for the online network. *Target network*, has the same architecture as online network and is not going to be trained. Instead, weights from the online network are periodically copied to target network.

# In[66]:


target_network = clone_model(online_network)
target_network.set_weights(online_network.get_weights())


# The target network uses past experience in the form of randomly selected records of the replay memory to predict targets for the online network: 
# 
# - Select a random minibatch from replay memory containing tuples $(\text{state},\text{action},\text{reward},\text{next_state})$
# 
# - For every tuple $(\text{state},\text{action},\text{reward},\text{next_state})$ from minibatch Q-value function $Q(\text{state},\text{action}\ |\ \theta_{\text{online}})$ is trained on predictions of $Q(\text{next_state}, a\ |\ \theta_\text{target})$ according to Bellman-type equation: 
# 
# $$y_\text{target} = \text{reward} + \gamma \cdot \max_a Q(\text{next_state}, a\ |\ \theta_\text{target})$$
# if the game continues and $$ y_\text{target} = \text{reward}$$ if the game has ended. 
# 
# Note that at this step predictions are made by the target network. This helps preventing situations when online network simultaneously predicts values and creates targets, which might potentially lead to instability of training process.
# 
# - For each record in the minibatch targets need to be calculated for only one specific $\text{action}$ output of online network. It is important to ignore all other outputs during optimization (calculating gradients). So, predictions for every record in the minibatch are calculated by online network first, then the values corresponding to the actually selected action are replaced with ones predicted by target network. 
# 
# ## Double DQN
# 
# Approach proposed in the previous section is called **DQN**-approach. 
# 
# DQN approach is very powerful and allows to train agents in very complex, very multidimentional environments.
# 
# However, [it is known](https://arxiv.org/abs/1509.06461) to overestimate q-values under certain conditions. 
# 
# Alternative approach proposed in the [same paper](https://arxiv.org/abs/1509.06461) is called **Double DQN**. 
# 
# Instead of taking action that maximizes q-value for target network, they pick an action that maximizes q-value for online network as an optimal one:
# 
# $$y_\text{target} = \text{reward} + \gamma \cdot Q\left(\text{next_state}, \arg\max_a Q\left(\text{next_state},a\ |\ \theta_\text{online}\right)\ |\ \theta_\text{target}\right).$$
# 

# Create folder for logs and trained weights:

# In[67]:


name = 'MsPacman_DQN'  # used in naming files (weights, logs, etc)
if not os.path.exists(name):
    os.makedirs(name)
    
weights_folder = os.path.join(name, 'weights')
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)


# ## Training DQN
# 
# First, define hyperparameters (Do not forget to change them before moving to cluster):

# In[68]:


#n_steps = 500000       # total number of training steps (= n_epochs)
#warmup = 5000
n_steps = 400000
warmup = 7000
                       # start training after warmup iterations
training_interval = 4  # period (in actions) between training steps
save_steps = int(n_steps/100)  # period (in training steps) between storing weights to file
copy_steps = 1000      # period (in training steps) between updating target_network weights
gamma = 0.9            # discount rate
#skip_start = 90       # skip the start of every game (it's just freezing time before game starts)
batch_size = 128       # size of minibatch that is taken randomly from replay memory every training step
double_dqn = True     # whether to use Double-DQN approach or simple DQN (see above)

# eps-greedy parameters: we slowly decrease epsilon from eps_max to eps_min in eps_decay_steps
eps_max = 1.0
eps_min = 0.05
eps_decay_steps = int(n_steps/2)

learning_rate = 0.0001


# Compile online-network with Adam optimizer, mean squared error loss and `mean_q` metric, which measures the maximum of predicted q-values averaged over samples from minibatch (we expect it to increase during training process).

# Use standard callbacks:

# In[69]:


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))
online_network.compile(optimizer=Adam(learning_rate), loss='mse', metrics=[mean_q])


# In[70]:


csv_logger = CSVLogger(os.path.join(name, 'log.csv'), append=True, separator=';')
tensorboard = TensorBoard(log_dir=os.path.join(name, 'tensorboard'), write_graph=False, write_images=False)


# In[71]:


import datetime
import pickle
start_time = datetime.datetime.now()
print(start_time)


# In[72]:


tensorboard


# In[73]:


csv_logger


# In[74]:


#replay_memory[0][4]
np.array(replay_memory).shape


# Next chunk of code explores the game, trains online network and periodically copies weights to target network as explained above.

# In[ ]:
from os import listdir
from os.path import isfile, join

weightsPath = root_dir+project_folder+'/'+name+'/weights'
search_string = 'online_weights_model'

def fetchLatest(path):
    f_split = []
    for f in listdir(path):
        if isfile(join(path, f)):
            if search_string in f:
                f_split.append(f.split("_")[3].split(".")[0])
    max_nbr = max(f_split)                
    return max_nbr
            
#onlyfiles = fetchLatest(weightsPath)

# counters:

step = 0          # training step counter (= epoch counter)
iteration = 0     # frames counter
episodes = 0      # game episodes counter
done = True       # indicator that env needs to be reset
all_rewards=[]
n_outputs = obs['possible_actions']
episode_scores = []  # collect total scores in this list and log it later
train_existing = False
weights_path = root_dir+project_folder+'/'+name+'/weights'
replmem_filename = 'replay_memory'


if train_existing:
  print("weights_path:", weights_path)
  start_time = datetime.datetime.now()
  print(start_time)
  
  # get the last processed step to restore weights
  lastPrcdStep = fetchLatest(weights_path)
  step = int(' '.join(lastPrcdStep))
  print("Using Loaded weights from step:" , lastPrcdStep)
  online_network.load_weights(os.path.join(weights_folder, 'online_weights_model_{}.h5'.format(step)))
  target_network.load_weights(os.path.join(weights_folder, 'target_weights_model_{}.h5'.format(step)))
  replmem_inpfile = open(replmem_filename,'rb')
  replmem_load = pickle.load(replmem_inpfile)
  replmem_inpfile.close()  
  replay_memory.append(replmem_load)
  print("Loaded replay Memory: ", replay_memory[0][4])
  #target_network.set_weights(online_network.get_weights())

  #n_steps = n_steps - step     # total number of training steps (= n_epochs)
  n_steps = 10000 + step    # total number of training steps (= n_epochs)
  warmup = 5000             # start training after warmup iterations
  gamma = 0.9               # discount rate
  batch_size = 128          # size of minibatch that is taken randomly from replay memory every training step
  double_dqn = True         # whether to use Double-DQN approach or simple DQN (see above)

    # eps-greedy parameters: we slowly decrease epsilon from eps_max to eps_min in eps_decay_steps
  eps_max = 0.8
  eps_min = 0.05
  eps_decay_steps = int(n_steps/4)
  learning_rate = 0.0001
  
print("Starting from step:", step)

while step < n_steps:

    if done:  # game over, restart it
        obs = env.reset()
        old_state = get_state(obs)
        episodes += 1

    # Online network evaluates what to do
    iteration += 1

    # Predict Online Network to calculate q_values
    q_values = online_network.predict(np.array([old_state]))[0]  # calculate q-values using online network

    # select epsilon (which linearly decreases over training steps):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    action = epsilon_greedy(q_values, epsilon, n_outputs)
    #print("Action:",action)
    # Play:

    obs = env.make_action(action)
    reward = obs['reward']
    done = obs['end_game']

    if done:
        episode_scores.append(obs['total_score'])

    next_state = get_state(env.get_obs())

    # Let's memorize what just happened

    all_rewards.append(reward)

    replay_memory.append((state, action, reward, next_state, done))
    state = next_state

    if iteration >= warmup and iteration % training_interval == 0:
        # learning branch
        step += 1

      # Encounter Bad value in replay memory while restoring from pickle file . So added temp fix to skip the bad record and generate sample with only good
      ###REMOVE## after fixing the pickle file
        while True:
          try:
            minibatch = random.sample(replay_memory, batch_size)
            replay_state = np.array([x[0] for x in minibatch])
            replay_action = np.array([x[1] for x in minibatch]) 
            replay_rewards = np.array([x[2] for x in minibatch])
            replay_next_state = np.array([x[3] for x in minibatch])
            replay_done = np.array([x[4] for x in minibatch], dtype=int)
            break
          except:
            x4 = [x[4] for x in minibatch]
            print(x4)
            pass
            #raise Exception("An exception occurred:")

        #print("replay_action :", replay_action)

        # calculate targets (see above for details)
        if double_dqn == False:
            # DQN
            target_for_action = replay_rewards + (1-replay_done) * gamma *                                     np.amax(target_network.predict(replay_next_state), axis=1)
        else:
            # Double DQN
            best_actions = np.argmax(online_network.predict(replay_next_state), axis=1)
            target_for_action = replay_rewards + (1-replay_done) * gamma *                                     target_network.predict(replay_next_state)[np.arange(batch_size), best_actions]
       
        target = online_network.predict(replay_state)  # targets coincide with predictions ...

        #since valid actions starts from 1. subtract 1 to align action 1 with index 0 and action 9 with index 8 to update target values
        target[np.arange(batch_size), replay_action-1] = target_for_action  #...except for targets with actions from replay
        
        # Train online network
        online_network.fit(replay_state, target, epochs=step, verbose=2, initial_epoch=step-1,
                           callbacks=[csv_logger, tensorboard])

        # Periodically copy online network weights to target network
        if step % copy_steps == 0:
            target_network.set_weights(online_network.get_weights())
        # And save weights
        if step % save_steps == 0:
            online_network.save_weights(os.path.join(weights_folder, 'online_weights_{}.h5f'.format(step)))
            target_network.save_weights(os.path.join(weights_folder, 'target_weights_{}.h5f'.format(step)))
            online_network.save(os.path.join(weights_folder, "online_weights_model_{}.h5".format(step)))
            target_network.save(os.path.join(weights_folder, "target_weights_model_{}.h5".format(step)))

            replmem_outfile = open(replmem_filename,'wb')
            pickle.dump(replay_memory,replmem_outfile)
            replmem_outfile.close()
            gc.collect()  # also clean the garbage


# In[ ]:


target.shape


# 

# In[ ]:


target[0]


# In[ ]:


#plt.plot(all_rewards)


# Save last weights:

# In[ ]:


online_network.save_weights(os.path.join(weights_folder, 'weights_train_last.h5f'))


# In[ ]:


# Dump all scores to txt-file
with open(os.path.join(name, 'episode_scores.txt'), 'w') as file:
    for item in episode_scores:
        file.write("{}\n".format(item))

print(episode_scores)


# Don't forget to check TensorBoard for fancy statistics on loss and metrics using in terminal
# 
# `tensorboard --logdir=tensorboard`
# 
# after navigating to the folder containing the created folder `tensorboard`: 

# In[ ]:


weights_folder


# Then visit http://localhost:6006/

# ## Testing model
# 
# Finally, create a function to evalutate the trained network. 
# Note that we still using $\varepsilon$-greedy strategy here to prevent an agent from getting stuck. 
# `test_dqn` returns a list with scores for the specified number of games.
# 
# **Below function is not used**. Used only for reference from Gym 

# In[ ]:


def test_dqn(env, n_games, model, nb_actions, skip_start, eps=0.05, render=False, sleep_time=0.01):
    scores = []
    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            state = obs
            q_values = model.predict(np.array([state]))[0]
            action = epsilon_greedy(q_values, eps, n_outputs)
            obs, reward, done, info = env.step(action)
            score += reward
            if render:
                env.render()
                time.sleep(sleep_time)
                if done:
                    time.sleep(1)
        scores.append(score)
    return scores


# **Testing model** 
# 
# Write a strategy using DQN, namely a function that takes an observation of PacmanGame and returns an action (integer from 1 to 9). Note that we use specific game parameters (such as number of monsters, diamonds, etc) stored in test_params.json.

# In[ ]:


def test_strategy(obs):
   state = get_state(obs)
   q_values = model.predict(np.array([state]))[0]
   eps = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
   action = epsilon_greedy(q_values, eps, obs['possible_actions'])
   return action


# In[ ]:


model = online_network.load_weights(os.path.join(weights_folder, 'weights_train_last.h5f'))


# In[ ]:


test(strategy=test_strategy, log_file='test_pacman_submit_DQN_log.json')


# In[ ]:


env.close()


# Results are pretty poor since the training was too short. 
# 
# Try to train DQN on a cluster. You might want to adjust some hyperparameters (increase `n_steps`, `warmup`, `copy_steps` and `eps_decay_steps`; gradually decrease learning rate during training, select appropriate `batch_size` to fit gpu memory, adjust `gamma`, switch on double dqn apporach and so on). 
# 
# You can even try to make the network deeper and/or use more than one observation as an input of neural network. For instance, using few consecutive game observations would definetely improve the results since they contain some helpful information such as monsters directions, etc. Turning off TensorBoard callback on a cluster would be a good idea too.
