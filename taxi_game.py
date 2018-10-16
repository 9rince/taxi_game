import gym
import os
from os import system
from time import sleep
import numpy as np
import random


env = gym.make('Taxi-v2').env
env.reset()
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()
print(env.P[328])

# solving the environment without reinforcement learning

env.s = 328
epochs = 0
penalties,rewards = 0,0

frames = []

done = False

while not done:

	action = env.action_space.sample()
	state , reward, done, info = env.step(action)

	if reward == -10:
		penalties += -1

	frames.append({
		'frame': env.render(mode='ansi'),
		'state': state,
		'action': action,
		'reward': reward
		})


	epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

def print_frames(frames):
	for i, frame in enumerate(frames):
		system('clear')
		print(frame['frame'].getvalue())
		print("Timestep: {}".format(i+1))
		print("State: {}".format(frame['state']))
		print("Action: {}".format(frame['action']))
		print("Reward: {}".format(frame['reward']))
		sleep(.1)
		
# print_frames(frames)


# solving the environment with Q learning
if os.path.isfile('qtable.npy'):
	q_table = np.load('qtable.npy')
	qble = 1
else:
	q_table = np.zeros([env.observation_space.n, env.action_space.n])
	qble = 0

#hyperparameters

alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []
epochs = 0
if not qble:
	for i in range(1,100000):
		
		state = env.reset()
		epochs,penalties,reward = 0,0,0
		done = False

		while not done:
			if random.uniform(0,1) < epsilon:
				action = env.action_space.sample()
			else:
				action = np.argmax(q_table[state])

			next_state,reward,done,info = env.step(action)
			old_value = q_table[state,action]
			next_max = np.max(q_table[next_state])
			new_value = (1-alpha)*old_value + alpha*(reward+ gamma*next_max)
			q_table[state,action] = new_value

			if reward == -10:
				penalties += 1

			state = next_state
			epochs += 1

		if i%100 ==0:
			system('clear')
			print('episode : {}'.format(i))

	np.save('qtable.npy',q_table)
	print('training finished')

# Evaluating the agent

total_epochs,total_penalties =0,0
episodes = 100

for k in range(episodes):
	state = env.reset()
	epochs,penalties,reward = 0,0,0
	done = False

	while not done:
		action  = np.argmax(q_table[state])
		state,reward,done,info = env.step(action)
		system('clear')
		print(k ,'\n')
		env.render()
		sleep(.2)
		

		if reward == -10:
			penalties += 1

		epochs += 1
	total_penalties += penalties
	total_epochs += epochs

print("Results after {} episodes:".format(episodes))
print("Average timesteps per episode: {}".format(total_epochs / episodes))
print("Average penalties per episode: {}".format(total_penalties / episodes))
