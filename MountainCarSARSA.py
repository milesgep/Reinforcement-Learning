import numpy as np
from numpy.random import normal
from numpy.random import uniform
import matplotlib.pyplot as plt

class MountainCarMDP:
	def __init__(self):
		self.reward = -1.0
		self.final_reward = 0.0

		self.boundary = [-1.2, 0.5]
		self.state = [-0.5, 0.0] # position, velocity
		self.norm_state = [0.5, 0.5]
		self.actions = [0, 1, 2]
		self.gamma = 1.0
		self.states = 2

	def give_MDP_info(self):
		return self.states, len(self.actions)#, self.final_state

	def change_state(self, action):

		self.state[1] = self.state[1] + 0.001*action - 0.0025*np.cos(3*self.state[0])
		if self.state[1] > 0.07:
			self.state[1] = 0.07
		elif self.state[1] < -0.07:
			self.state[1] = -0.07

		self.state[0] = self.state[0] + self.state[1]

		# normalize the state
		self.norm_state[0] = (self.state[0] + 1.2) / 1.7
		self.norm_state[1] = (self.state[1] + 0.07) / 0.14

		if(self.state[0] >= self.boundary[1]):
			self.state = [self.boundary[1], 0]
			self.norm_state = [1, 0.5]
			return self.norm_state, self.final_reward
		elif(self.state[0] <= self.boundary[0]):
			self.state = [self.boundary[0], 0]
			self.norm_state = [0, 0.5]

		return self.norm_state, self.reward


class Agent:
	def __init__(self, action_space, gamma, q_table, epsilon, alpha, lambd):
		self.action_space = action_space
		self.sumOfRewards = 0.0
		self.gamma = gamma
		self.q_table = q_table
		self.epsilon = epsilon
		self.alpha = alpha
		self.lambd = lambd

	def make_softmax_move(self, state, weight_matrix):
		left_state = np.matmul(weight_matrix[0].T, np.cos(np.pi * np.dot(self.q_table, (state + [1, 0, 0]))))
		neutral_state = np.matmul(weight_matrix[1].T, np.cos(np.pi * np.dot(self.q_table, (state + [0, 1, 0]))))
		right_state = np.matmul(weight_matrix[2].T, np.cos(np.pi * np.dot(self.q_table, (state + [0, 0, 1]))))

		table = np.array([left_state, neutral_state, right_state])

		prob_array = np.exp(self.epsilon * table - np.max(self.epsilon * table)) / np.sum(np.exp(self.epsilon*table - np.max(self.epsilon * table)))

		move = np.random.choice(a=[-1, 0, 1], size=1, p=prob_array)[0]
		if move == -1:
			action_matrix = [1, 0, 0]
		elif move == 0:
			action_matrix = [0, 1, 0]
		else:
			action_matrix = [0, 0, 1]
		return move, action_matrix

	def make_greedy_move(self, state, weight_matrix):
		left_state = np.matmul(weight_matrix[0].T, np.cos(np.pi * np.dot(self.q_table, (state + [1, 0, 0]))))
		neutral_state = np.matmul(weight_matrix[1].T, np.cos(np.pi * np.dot(self.q_table, (state + [0, 1, 0]))))
		right_state = np.matmul(weight_matrix[2].T, np.cos(np.pi * np.dot(self.q_table, (state + [0, 0, 1]))))

		if left_state == np.max([left_state, neutral_state, right_state]):
			action_index = 0
			action_matrix = [1, 0, 0]
		elif neutral_state == np.max([left_state, neutral_state, right_state]):
			action_index = 1
			action_matrix = [0, 1, 0]
		else:
			action_index = 2
			action_matrix = [0, 0, 1]

		prob_array = [self.epsilon/3.0, self.epsilon/3.0, self.epsilon/3.0]
		prob_array[action_index] = 1.0 - self.epsilon + (self.epsilon/3.0)
		move = np.random.choice(a=[-1, 0, 1], size=1, p=prob_array)[0]
		return move, action_matrix

	def response(self, reward):
		self.sumOfRewards += reward
		return
		
	def give_policy(self):
		return self.policy

	def run(self, MDP, weight_matrix, greedy):
		position = [0.412, 0.5]
		eligibility = 0
		if greedy:
			action, action_matrix = self.make_greedy_move(position, weight_matrix)
		else:
			action, action_matrix = self.make_softmax_move(position, weight_matrix)
		time_step = 0
		reward = -1
		while reward != MDP.final_reward:
			next_position, reward = MDP.change_state(action)
			self.response(reward)
			if greedy:
				next_action, next_action_matrix = self.make_greedy_move(next_position, weight_matrix)
			else:
				next_action, next_action_matrix = self.make_softmax_move(next_position, weight_matrix)
			weight_matrix, eligibility = Lambda_update(position, action, action_matrix, next_position, next_action, next_action_matrix, reward, self.gamma, self.alpha, self.lambd, eligibility, self.q_table, weight_matrix)
			position = next_position.copy()
			action_matrix = next_action_matrix.copy()
			action = next_action
			time_step += 1
			if time_step == 500:
				break

		return weight_matrix, self.sumOfRewards



def evaluate(number_of_episodes, fourier_basis, gamma, alpha, epsilon, lambd, greedy):
	q_table = make_phi(fourier_basis)
	expected_returns = []
	weight_matrix = np.zeros((3, len(q_table)))
	count = 0
	for episode in range(number_of_episodes):
		MDP = MountainCarMDP()
		state_space, action_space = MDP.give_MDP_info()
		episode_agent = Agent(action_space, gamma, q_table, epsilon, alpha, lambd)
		weight_matrix, expected_return = episode_agent.run(MDP, weight_matrix, greedy)
		expected_returns.append(expected_return)
		if greedy:
			epsilon *= 0.95
	#print(np.mean(expected_returns), max(expected_returns))
	#print(expected_returns)
		
	return expected_returns

def Lambda_update(state, action, action_matrix, new_state, new_action, new_action_matrix, reward, gamma, alpha, lambd, eligibility, phi_s, weight_matrix):
	eligibility = lambd*eligibility + np.cos(np.pi * np.dot(phi_s, state + action_matrix))

	state_phi_s = np.matmul(weight_matrix[action+1].T, np.cos(np.pi * np.dot(phi_s, state + action_matrix)))
	new_state_phi_s = np.matmul(weight_matrix[new_action+1].T, np.cos(np.pi * np.dot(phi_s, new_state + new_action_matrix)))

	error = reward + gamma*new_state_phi_s - state_phi_s

	weight_matrix[action+1] = weight_matrix[action+1].T + (error*alpha*eligibility).T
	return weight_matrix, eligibility

def make_phi(fourier_basis):
	phi_s = []
	for basis_1 in range(fourier_basis+1):
		for basis_2 in range(fourier_basis+1):
			for basis_3 in range(fourier_basis+1):
				for basis_4 in range(fourier_basis+1):
					for basis_5 in range(fourier_basis+1):
							phi_s.append([basis_1, basis_2, basis_3, basis_4, basis_5])

	phi_s = np.array(phi_s)
	return phi_s

def main():

	plt.figure(figsize=(18, 16), dpi=100, facecolor='w', edgecolor='k')

	number_of_episodes = 100
	gamma = 1.0
	
	trials = 100

	greedy = False
	lambd = 0
	epsilon = 5
	alpha = 1e-4

	fourier_basis = 5


	#for alpha in alphas:
		#for epsilon in epsilons:
	#	for lambd in lambdas:
	all_trial_data = np.zeros((trials, number_of_episodes))
	print("number of trials =", trials, "alpha =", alpha, "lambd =", lambd)
	for trial in range(trials):
		trial_data = evaluate(number_of_episodes, fourier_basis, gamma, alpha, epsilon, lambd, greedy)
		trial_data = np.array(trial_data).flatten()
		all_trial_data[trial] = trial_data

	#plt.plot(range(number_of_episodes), np.mean(all_trial_data, axis=0), color='blue', label="Epsilon Greedy")



	#plt.errorbar(range(len(all_trial_data[0])), np.mean(all_trial_data, axis=0), yerr=np.std(all_trial_data, axis=0), ecolor='yellow', fmt='o')
	plt.errorbar(range(len(all_trial_data[0])), np.mean(all_trial_data, axis=0), yerr=np.std(all_trial_data, axis=0), color='blue', ecolor='green', fmt='o')
	#plt.plot(range(number_of_episodes), np.mean(all_trial_data, axis=0), color='blue', label="Softmax")
	plt.title('Mountain Car Sarsa')
	plt.xlabel('Episodes', fontsize=18)
	plt.ylabel('Expected Return', fontsize=18)
	plt.legend()
	plt.show()
  
if __name__== "__main__":
  main()