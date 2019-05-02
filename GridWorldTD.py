import numpy as np
from numpy.random import normal
from numpy.random import uniform
import matplotlib.pyplot as plt

class GridWorldMDP:
	def __init__(self):
		self.grid = [[ 1,  2,  3,  4, 5],
					[ 6,  7,  8,  9, 10],
					[11, 12, -1, 13, 14],
					[15, 16, -1, 17, 18],
					[19, 20, 21, 22, 23]]
		self.state = [0, 0]
		self.position = 1
		self.actions = [0, 1, 2, 3] # left, up, right, down
		self.states = 23
		#self.final_state = 23
		self.final_state = [4, 4]
		self.reward = [[ 0, 0,   0, 0, 0],
					   [ 0, 0,   0, 0, 0],
					   [ 0, 0,   0, 0, 0],
					   [ 0, 0,   0, 0, 0],
					   [ 0, 0, -10, 0, 10]]

	def give_MDP_info(self):
		return self.states, len(self.actions), self.final_state

	def change_state(self, action):

		if action == -1:
			self.position = self.final_state
			return self.final_state, 0

		chance = uniform(0, 1)
		if chance < 0.05:
			action = action - 1
		elif chance < 0.10:
			action = action + 1
		elif chance < 0.20:
			action = -10 # stay, no action


		if action == 0:
			self.state[1] -= 1
		elif action == 1:
			self.state[0] -= 1
		elif action == 2:
			self.state[1] += 1
		elif action == -1 or action == 3:
			self.state[0] += 1

		# Check for illegal move
		if self.state[0] < 0:
			self.state[0] = 0
		elif self.state[0] > 4:
			self.state[0] = 4
		elif self.state[1] < 0:
			self.state[1] = 0
		elif self.state[1] > 4:
			self.state[1] = 4
		elif self.state[1] == 2 and (self.state[0] == 2 or self.state[0] == 3):
			if action == 0:
				self.state[1] = 3
			elif action == 1:
				self.state[0] = 4
			elif action == 2:
				self.state[1] = 1
			elif action == -1 or action == 3:
				self.state[0] = 1

		self.position = self.grid[self.state[0]][self.state[1]]
		reward = self.reward[self.state[0]][self.state[1]]
		return self.state, reward


class Agent:
	def __init__(self, action_space, final_state, gamma = 1.0):
		self.action_space = action_space
		self.sumOfRewards = 0
		self.final_state = final_state
		self.gamma = gamma


	def make_random_move(self):
		move = np.random.choice(a=[0, 1, 2, 3], size=1, p=[0.25, 0.25, 0.25, 0.25])
		return move


	def run(self, MDP, state_values, gamma, alpha, train):
		td_error = []
		position = []
		while self.final_state != MDP.state:
			previous_position = position.copy()
			if train:
				position, reward = MDP.change_state(self.make_random_move())
				if previous_position:
					state_values[previous_position[0]][previous_position[1]] = TD_update(state_values, previous_position, position, reward, gamma, alpha)
			else:
				position, reward = MDP.change_state(self.make_random_move())
				if previous_position:
					td_error.append((reward + (gamma * state_values[position[0]][position[1]]) - state_values[previous_position[0]][previous_position[1]])**2)
				
		return state_values, td_error


def evaluate(state_values, number_of_episodes, gamma = 0.9, alpha = 0.01, train = True):
	all_td_errors = []
	for episode in range(number_of_episodes):
		MDP = GridWorldMDP()
		state_space, action_space, final_state = MDP.give_MDP_info()
		episode_agent = Agent(action_space, final_state, gamma)
		state_values, td_errors = episode_agent.run(MDP, state_values, gamma, alpha, train)
		if not train:
			all_td_errors.extend(td_errors)
	all_td_errors = np.array(all_td_errors)
	print(state_values)
	if train:
		return state_values
	else:
		return state_values, all_td_errors



def TD_update(state_values, state, new_state, reward, gamma, alpha):
	value = state_values[state[0]][state[1]] + (alpha*(reward + (gamma * state_values[new_state[0]][new_state[1]]) - state_values[state[0]][state[1]]))
	return value


def main():

	plt.figure(figsize=(18, 16), dpi=100, facecolor='w', edgecolor='k')

	alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
	plot_alphas = np.log10(alphas)
	ind = np.arange(len(alphas))
	number_of_episodes = 100
	gamma = 0.9

	MSE_td = []

	print("number of episodes:", number_of_episodes)

	for alpha in alphas:
		state_values = np.zeros((5,5))
		state_values = evaluate(state_values=state_values, number_of_episodes=number_of_episodes, gamma=gamma, alpha=alpha, train = True)
		state_values, td_errors = evaluate(state_values=state_values, number_of_episodes=number_of_episodes, gamma=gamma, alpha=alpha, train = False)
		MSE_td.append(np.mean(td_errors))
		print(alpha, len(td_errors), np.mean(td_errors))


	plt.plot(plot_alphas, MSE_td, label="GridWorld")

	plt.xlabel(r"$log_{10}{(Step Size)}$", fontsize=18)
	plt.ylabel('Mean Squared TD Error', fontsize=18)
	plt.legend()
	plt.show()
  
if __name__== "__main__":
  main()