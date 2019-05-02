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
		self.final_state = 23
		#self.final_state = [4, 4]
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
		return self.position, reward


class Agent:
	def __init__(self, action_space, final_state, gamma, q_table, epsilon, alpha, lambd):
		self.action_space = action_space
		self.sumOfRewards = 0.0
		self.final_state = final_state
		self.gamma = gamma
		self.q_table = q_table
		self.epsilon = epsilon
		self.alpha = alpha
		self.lambd = lambd
		self.final_time = 1000


	def response(self, reward):
		self.sumOfRewards += ((self.gamma ** (self.time_step-1)) * reward)
		return

	def make_softmax_move(self, state):
		prob_array = np.exp(self.epsilon * self.q_table[state-1] - np.max(self.epsilon * self.q_table[state-1])) / np.sum(np.exp(self.epsilon*self.q_table[state-1] - np.max(self.epsilon * self.q_table[state-1])))

		move = np.random.choice(a=[0, 1, 2, 3], size=1, p=prob_array)[0]
		return move

	def make_greedy_move(self, state):
		if self.q_table[state-1].count(self.q_table[state-1][0]) == len(self.q_table[state-1]):
			action = np.random.choice(a=[0, 1, 2, 3], size=1, p=[0.25, 0.25, 0.25, 0.25])[0]
		else:
			action = np.argmax(self.q_table[state-1])

		prob_array = [self.epsilon/4.0, self.epsilon/4.0, self.epsilon/4.0, self.epsilon/4.0]
		prob_array[action] = 1.0 - self.epsilon + (self.epsilon/4.0)
		move = np.random.choice(a=[0, 1, 2, 3], size=1, p=prob_array)[0]
		return move

	def run(self, MDP, greedy):
		position = 1
		eligibility = np.zeros((len(self.q_table), len(self.q_table[0])))
		self.time_step = 1
		if greedy:
			action = self.make_greedy_move(position)
		else:
			action = self.make_softmax_move(position)

		while self.final_state != MDP.position:
			next_position, reward = MDP.change_state(action)
			self.response(reward)
			self.time_step += 1
			if greedy:
				next_action = self.make_greedy_move(next_position)
			else:
				next_action = self.make_softmax_move(next_position)
			
			self.q_table[position-1][action], eligibility = Lambda_update(position-1, action, next_position-1, next_action, reward, self.gamma, self.alpha, self.lambd, eligibility, self.q_table)
			position = next_position
			action = next_action

			if self.time_step == self.final_time:
				break

		return self.q_table, self.sumOfRewards


def evaluate(number_of_episodes, gamma, alpha, epsilon, lambd, greedy):
	q_table = make_QTable()
	if not greedy:
		q_table = np.array(q_table)
	expected_returns = []
	count = 0
	for episode in range(number_of_episodes):
		MDP = GridWorldMDP()
		state_space, action_space, final_state = MDP.give_MDP_info()
		episode_agent = Agent(action_space, final_state, gamma, q_table, epsilon, alpha, lambd)
		q_table, expected_return = episode_agent.run(MDP, greedy)
		if expected_return > 4.5:
			count+=1
		expected_returns.append(expected_return)
		# epsilon decay
		if greedy:
			epsilon *= 0.9
	#if count > 15:
	#	print(count)
	return q_table, expected_returns


def Lambda_update(state, action, new_state, new_action, reward, gamma, alpha, lambd, eligibility, phi_s):
	eligibility[state][action] = lambd*eligibility[state][action] + 1

	action_value = phi_s[state][action]
	next_action_value = phi_s[new_state][new_action]

	error = reward + gamma*next_action_value - action_value

	value = phi_s[state][action] + (error*alpha*eligibility[state][action])
	return value, eligibility

def make_QTable():
	q_table = []
	for i in range(23):
		q_table.append([0.0, 0.0, 0.0, 0.0])

	return q_table

def update_QTable(q_table, state, action, value):
	q_table[state, action] = value
	return q_table


def main():

	plt.figure(figsize=(18, 16), dpi=100, facecolor='w', edgecolor='k')

	number_of_episodes = 100
	gamma = 0.9

	trials = 500

	#alphas = np.random.uniform(0.00000, 0.1, 5)
	#epsilons = np.random.uniform(0.00000, 0.1, 5)
	#lambdas = np.random.uniform(0.00000, 0.1, 5)

	greedy = True

	# best values so far
	epsilon = 0.09893299107250408 
	alpha = 0.03427476426325967 
	lambd = 0.04913153868598627

	#for alpha in alphas:
	#	for epsilon in epsilons:
	#		for lambd in lambdas:
				#all_trial_data = []
	all_trial_data = np.zeros((trials, number_of_episodes))
	print("number of episodes =", number_of_episodes, "number of trials =", trials, "epsilon =", epsilon, "alpha =", alpha, "lambd =", lambd)
	for trial in range(trials):
		q_table, trial_data = evaluate(number_of_episodes, gamma, alpha, epsilon, lambd, greedy)
		trial_data = np.array(trial_data).flatten()
		all_trial_data[trial] = trial_data
			

	plt.errorbar(range(len(all_trial_data[0])), np.mean(all_trial_data, axis=0), yerr=np.std(all_trial_data, axis=0), color='blue', ecolor='green', fmt='o')
	#plt.plot(range(number_of_episodes), np.mean(all_trial_data, axis=0), color='blue', label="Epsilon Greedy")
	plt.xlabel('Episodes', fontsize=18)
	plt.ylabel('Expected Return', fontsize=18)
	plt.legend()
	plt.show()
			#plt.gcf().clear()
			#plt.figure(figsize=(18, 16), dpi=100, facecolor='w', edgecolor='k')
  
if __name__== "__main__":
  main()