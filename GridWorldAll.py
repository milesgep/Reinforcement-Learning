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


### CEM ###

class CEM_Agent:
	def __init__(self, action_space, final_state, policy, gamma = 1.0):
		self.action_space = action_space
		self.sumOfRewards = 0
		self.final_state = final_state
		self.time_step = 0
		self.final_time = 100#00
		self.gamma = gamma
		self.policy = self.stabilize_and_convert_policy(policy)

	def stabilize_and_convert_policy(self, policy):
		new_policy = np.zeros(policy.shape)
		for i in range(int(len(policy)/self.action_space)):
			max_val = np.max(policy[i*self.action_space:i*self.action_space+4])
			new_policy[i*self.action_space] = policy[i*self.action_space] - max_val
			new_policy[i*self.action_space+1] = policy[i*self.action_space+1] - max_val
			new_policy[i*self.action_space+2] = policy[i*self.action_space+2] - max_val
			new_policy[i*self.action_space+3] = policy[i*self.action_space+3] - max_val

			denominator = np.exp(new_policy[i*self.action_space]) + np.exp(new_policy[i*self.action_space + 1]) + np.exp(new_policy[i*self.action_space + 2]) + np.exp(new_policy[i*self.action_space + 3])
			new_policy[i*self.action_space] = np.exp(new_policy[i*self.action_space])/denominator
			new_policy[i*self.action_space+1] = np.exp(new_policy[i*self.action_space+1])/denominator
			new_policy[i*self.action_space+2] = np.exp(new_policy[i*self.action_space+2])/denominator
			new_policy[i*self.action_space+3] = np.exp(new_policy[i*self.action_space+3])/denominator

		return new_policy

	def make_policy_move(self, state):
		if self.terminate() == -1:
			return -1
		action = uniform(0, 1)
		index = state-1

		if action < self.policy[index*self.action_space]:
			return 0
		elif action < (self.policy[index*self.action_space] + self.policy[index*self.action_space + 1]):
			return 1
		elif action < (self.policy[index*self.action_space] + self.policy[index*self.action_space + 1] + self.policy[index*self.action_space + 2]):
			return 2
		else:
			return 3


	def response(self, state, reward):
		self.sumOfRewards += ((self.gamma ** (self.time_step-1)) * reward)
		return
		

	def give_policy(self):
		return self.policy

	def terminate(self):
		self.time_step += 1
		if self.time_step == self.final_time:
			return -1


	def run(self, MDP):
		while self.final_state != MDP.position:
			position, reward = MDP.change_state(self.make_policy_move(MDP.position))
			self.response(position, reward)
		return self.sumOfRewards



def CEM_evaluate(policy, number_of_episodes, gamma):
	expected_returns = []
	for episode in range(number_of_episodes):
		MDP = GridWorldMDP()
		state_space, action_space, final_state = MDP.give_MDP_info()
		episode_agent = CEM_Agent(action_space, final_state, policy, gamma)
		episode_return = episode_agent.run(MDP)
		expected_returns.append(episode_return)
	approx_j = np.mean(expected_returns)

	return approx_j, expected_returns

### Q-learning ###

class Q_Agent:
	def __init__(self, action_space, final_state, gamma, q_table, epsilon, alpha):
		self.action_space = action_space
		self.sumOfRewards = 0.0
		self.final_state = final_state
		self.gamma = gamma
		self.q_table = q_table
		self.epsilon = epsilon
		self.alpha = alpha
		self.final_time = 1000

	def terminate(self):
		if self.time_step == self.final_time:
			return -1

	def response(self, reward):
		self.sumOfRewards += ((self.gamma ** (self.time_step-1)) * reward)
		return

	def make_random_move(self):
		move = np.random.choice(a=[0, 1, 2, 3], size=1, p=[0.25, 0.25, 0.25, 0.25])
		return move

	def make_policy_move(self, state):
		if self.terminate() == -1:
			return -1
		# subtract one to state since it is 1 to 23
		if self.q_table[state-1].count(self.q_table[state-1][0]) == len(self.q_table[state-1]):
			action = np.random.choice(a=[0, 1, 2, 3], size=1, p=[0.25, 0.25, 0.25, 0.25])[0]
		else:
			action = np.argmax(self.q_table[state-1])

		prob_array = [self.epsilon/4.0, self.epsilon/4.0, self.epsilon/4.0, self.epsilon/4.0]
		prob_array[action] = 1.0 - self.epsilon + (self.epsilon/4.0)
		move = np.random.choice(a=[0, 1, 2, 3], size=1, p=prob_array)[0]
		return move

	def run(self, MDP):
		position = 1
		self.time_step = 1
		while self.final_state != MDP.position:
			previous_position = position
			action = self.make_policy_move(position)
			if action == -1:
				break

			position, reward = MDP.change_state(action)
			self.q_table[previous_position-1][action] = Q_update(self.q_table, previous_position-1, action, position-1, reward, self.gamma, self.alpha)
			self.response(reward)
			self.time_step += 1

		return self.q_table, self.sumOfRewards


def Q_evaluate(number_of_episodes, gamma, alpha, epsilon):
	q_table = make_GW_QTable()
	expected_returns = []
	for episode in range(number_of_episodes):
		MDP = GridWorldMDP()
		state_space, action_space, final_state = MDP.give_MDP_info()
		episode_agent = Q_Agent(action_space, final_state, gamma, q_table, epsilon, alpha)
		q_table, expected_return = episode_agent.run(MDP)

		# epsilon decay
		epsilon *= 0.9
		expected_returns.append(expected_return)
		
	return q_table, expected_returns


def Q_update(q_table, state, action, new_state, reward, gamma, alpha):
	value = q_table[state][action] + (alpha*(reward + (gamma * np.max(q_table[new_state])) - q_table[state][action]))
	return value

def make_GW_QTable():
	q_table = []
	for i in range(23):
		q_table.append([0.0, 0.0, 0.0, 0.0])

	return q_table

def update_QTable(q_table, state, action, value):
	q_table[state, action] = value
	return q_table


### SARSA ###

class SARSA_Agent:
	def __init__(self, action_space, final_state, gamma, q_table, epsilon, alpha):
		self.action_space = action_space
		self.sumOfRewards = 0.0
		self.final_state = final_state
		self.gamma = gamma
		self.q_table = q_table
		self.epsilon = epsilon
		self.alpha = alpha
		self.final_time = 1000

	def terminate(self):
		if self.time_step == self.final_time:
			return -1

	def response(self, reward):
		self.sumOfRewards += ((self.gamma ** (self.time_step-1)) * reward)
		return

	def make_random_move(self):
		move = np.random.choice(a=[0, 1, 2, 3], size=1, p=[0.25, 0.25, 0.25, 0.25])
		return move

	def make_policy_move(self, state):
		if self.terminate() == -1:
			return -1
		# subtract one to state since it is 1 to 23
		if self.q_table[state-1].count(self.q_table[state-1][0]) == len(self.q_table[state-1]):
			action = np.random.choice(a=[0, 1, 2, 3], size=1, p=[0.25, 0.25, 0.25, 0.25])[0]
		else:
			action = np.argmax(self.q_table[state-1])
		prob_array = [self.epsilon/4.0, self.epsilon/4.0, self.epsilon/4.0, self.epsilon/4.0]
		prob_array[action] = 1.0 - self.epsilon + (self.epsilon/4.0)
		move = np.random.choice(a=[0, 1, 2, 3], size=1, p=prob_array)[0]
		return move

	def run(self, MDP):
		position = 1
		self.time_step = 1
		action = self.make_policy_move(position)

		while self.final_state != MDP.position:
			next_position, reward = MDP.change_state(action)
			self.response(reward)
			self.time_step += 1
			next_action = self.make_policy_move(next_position)

			if next_action == -1:
				break
			
			self.q_table[position-1][action] = S_update(self.q_table, position-1, action, next_position-1, next_action, reward, self.gamma, self.alpha)
			position = next_position
			action = next_action

		return self.q_table, self.sumOfRewards


def evaluate(number_of_episodes, gamma, alpha, epsilon):
	q_table = make_GW_QTable()
	expected_returns = []
	#count = 0
	for episode in range(number_of_episodes):
		MDP = GridWorldMDP()
		state_space, action_space, final_state = MDP.give_MDP_info()
		episode_agent = Agent(action_space, final_state, gamma, q_table, epsilon, alpha)
		q_table, expected_return = episode_agent.run(MDP)
		#if expected_return > 4.5:
		#	count+=1
		expected_returns.append(expected_return)

		# epsilon decay
		epsilon *= 0.95
	#print("count =", count)
	#print(expected_returns)
	#for i in range(len(q_table)):
	#	print(q_table[i])
		
	return q_table, expected_returns


def S_update(q_table, state, action, new_state, new_action, reward, gamma, alpha):
	value = q_table[state][action] + (alpha*(reward + (gamma * (q_table[new_state][new_action])) - q_table[state][action]))
	return value

def main():


	GridWorld = GridWorldMDP()
	state_space, action_space, _ = GridWorld.give_MDP_info()
	plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

	### CEM ###

	sigma = 0.3
	epsilon = 0.125
	population = 20
	elite_population = 10
	number_of_episodes = 20
	gamma = 0.9


	iterations = 1000
	trials = 500
	all_trial_data = np.zeros((trials, iterations*population*number_of_episodes))

	print("sigma:", sigma, "epsilon:", epsilon, "population:", population, "elite_population:", elite_population, "number of episodes:", number_of_episodes, "iterations", iterations)
	for trial in range(trials):
		mean_policy = np.zeros((state_space, action_space)).flatten()
		covariance_matrix = sigma * np.identity(mean_policy.shape[0])
		trial_data = []
		for iteration in range(iterations):
			policies = []
			k_policies = np.random.multivariate_normal(mean_policy, covariance_matrix, population)
			for pop in range(population):
				obj_function, expected_returns = CEM_evaluate(k_policies[pop], number_of_episodes, gamma)
				policies.append((obj_function, k_policies[pop]))
				trial_data.append(expected_returns)
			policies.sort(key=lambda x: x[0])
			mean_policy = np.zeros(mean_policy.shape)
			for j_hat, pol in policies[-elite_population:]:
				mean_policy += pol
			mean_policy /= elite_population

			covariance_matrix = np.zeros(covariance_matrix.shape)
			for val in policies[-elite_population:][1]:
				best_policies = np.array(val - mean_policy).reshape(mean_policy.shape[0], 1)
				covariance_matrix += np.dot(best_policies,best_policies.T)
			covariance_matrix = (epsilon*np.identity(covariance_matrix.shape[0]) + covariance_matrix)/(epsilon+elite_population)


			
		trial_data = np.array(trial_data).flatten()
		all_trial_data[trial] = trial_data

		if trial % 10 == 0:
			print('trial %d' % (trial))

	### Q-learning ###		

	epsilon = 0.8566655155975921 
	alpha = 0.004247968034180764

	Q_all_trial_data = np.zeros((trials, number_of_episodes))
	print("number of episodes =", number_of_episodes, "number of trials =", trials, "epsilon =", epsilon, "alpha =", alpha)
	for trial in range(trials):
		q_table, trial_data = Q_evaluate(number_of_episodes, gamma, alpha, epsilon)
		trial_data = np.array(trial_data).flatten()
		Q_all_trial_data[trial] = trial_data


	### SARSA ###
	epsilon = 0.29809581751426917 
	alpha = 0.015068279887021916

	SARSA_all_trial_data = np.zeros((trials, number_of_episodes))
	print("number of episodes =", number_of_episodes, "number of trials =", trials, "epsilon =", epsilon, "alpha =", alpha)
	for trial in range(trials):
		q_table, trial_data = evaluate(number_of_episodes, gamma, alpha, epsilon)
		trial_data = np.array(trial_data).flatten()
		SARSA_all_trial_data[trial] = trial_data

	plt.errorbar(range(len(all_trial_data[0])), np.mean(all_trial_data, axis=0), yerr=np.std(all_trial_data, axis=0), ecolor='yellow', fmt='o')
	plt.errorbar(range(len(all_trial_data[0])), np.mean(Q_all_trial_data, axis=0), yerr=np.std(Q_all_trial_data, axis=0), ecolor='green', fmt='o')
	plt.errorbar(range(len(all_trial_data[0])), np.mean(SARSA_all_trial_data, axis=0), yerr=np.std(SARSA_all_trial_data, axis=0), ecolor='green', fmt='o')

	plt.xlabel('Episodes', fontsize=18)
	plt.ylabel('Expected Return', fontsize=18)
	plt.show()
	plt.savefig('GW.png')
	plt.gcf().clear()
  
if __name__== "__main__":
  main()