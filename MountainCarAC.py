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
		self.norm_state = [0.412, 0.5]
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
	def __init__(self, action_space, gamma, phi, epsilon, alpha1, alpha2, lambd):
		self.action_space = action_space
		self.sumOfRewards = 0.0
		self.gamma = gamma
		self.phi = phi
		self.epsilon = epsilon
		self.alpha1 = alpha1
		self.alpha2 = alpha2
		self.lambd = lambd

	def make_softmax_move(self, state, weight_matrix):
		left_state = np.matmul(weight_matrix[0].T, np.cos(np.pi * np.dot(self.phi, state)))
		neutral_state = np.matmul(weight_matrix[1].T, np.cos(np.pi * np.dot(self.phi, state)))
		right_state = np.matmul(weight_matrix[2].T, np.cos(np.pi * np.dot(self.phi, state)))

		table = np.array([left_state, neutral_state, right_state])

		prob_array = np.exp(table - np.max(table)) / np.sum(np.exp(table - np.max(table)))
		#prob_array = np.exp(self.epsilon * table) / np.sum(np.exp(self.epsilon*table))

		move = np.random.choice(a=[-1, 0, 1], size=1, p=prob_array)[0]
		return move

	def softmax_derivative(self, state, weight_matrix, action):
		left_state = np.matmul(weight_matrix[0].T, np.cos(np.pi * np.dot(self.phi, state)))
		neutral_state = np.matmul(weight_matrix[1].T, np.cos(np.pi * np.dot(self.phi, state)))
		right_state = np.matmul(weight_matrix[2].T, np.cos(np.pi * np.dot(self.phi, state)))

		table = np.array([left_state, neutral_state, right_state])

		prob_array = -np.exp(table - np.max(table)) / np.sum(np.exp(table - np.max(table)))
		#prob_array = np.exp(self.epsilon * table) / np.sum(np.exp(self.epsilon*table))

		prob_array[action+1] = 1 + prob_array[action+1]
		#prob_array = np.array([weight_matrix[0]*prob_array[0], weight_matrix[1]*prob_array[1], weight_matrix[2]*prob_array[2]])
		prob_array = np.outer(np.cos(np.pi * np.dot(self.phi, state)), prob_array)
		#move = np.random.choice(a=[-1, 0, 1], size=1, p=prob_array)[0]
		return prob_array


	def response(self, reward):
		self.sumOfRewards += reward
		return
		
	def give_policy(self):
		return self.policy

	def run(self, MDP, weight_matrix, theta, greedy):
		position = [0.412, 0.5]
		eligibility_v = 0
		eligibility_theta = 0
		time_step = 0
		reward = -1
		while reward != MDP.final_reward:
			previous_position = position.copy()
			action = self.make_softmax_move(position, theta)
			position, reward = MDP.change_state(action)
			#Critic Update
			weight_matrix, eligibility_v, error = Lambda_update(previous_position, action, position, reward, self.gamma, self.alpha1, self.lambd, eligibility_v, self.phi, weight_matrix)
			
			# Actor Update
			eligibility_theta = self.gamma*self.lambd*eligibility_theta + self.softmax_derivative(previous_position, theta, action)
			theta = theta + (self.alpha2*error*eligibility_theta).T

			self.response(reward)
			time_step += 1
			if time_step == 500:
				break
		return weight_matrix, theta, self.sumOfRewards



def evaluate(number_of_episodes, fourier_basis, gamma, alpha1, alpha2, epsilon, lambd, greedy):
	phi = make_phi(fourier_basis)
	weight_matrix = np.zeros((len(phi)))
	theta = np.zeros((3, len(phi)))
	expected_returns = []
	for episode in range(number_of_episodes):
		MDP = MountainCarMDP()
		state_space, action_space = MDP.give_MDP_info()
		episode_agent = Agent(action_space, gamma, phi, epsilon, alpha1, alpha2, lambd)
		weight_matrix, theta, expected_return = episode_agent.run(MDP, weight_matrix, theta, greedy)
		expected_returns.append(expected_return)
		#if expected_return == 1010.0:
		#	count+=1
	#if max(expected_returns) > -500:
	print(np.mean(expected_returns), max(expected_returns))
		
	return expected_returns

def Lambda_update(state, action, new_state, reward, gamma, alpha, lambd, eligibility, phi_s, weight_matrix):
	eligibility = lambd*eligibility + np.cos(np.pi * np.dot(phi_s, state))

	state_phi_s = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(phi_s, state)))
	new_state = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(phi_s, new_state)))

	error = reward + gamma*new_state - state_phi_s

	weight_matrix = weight_matrix.T + (error*alpha*eligibility).T
	return weight_matrix.T, eligibility, error

def make_phi(fourier_basis):
	phi_s = []
	for basis_1 in range(fourier_basis+1):
		for basis_2 in range(fourier_basis+1):
			phi_s.append([basis_1, basis_2])

	phi_s = np.array(phi_s)
	return phi_s

def make_phi_actor(fourier_basis):
	phi_s = []
	for basis_1 in range(fourier_basis+1):
		for basis_2 in range(fourier_basis+1):
			for basis_3 in range(fourier_basis+1):
				for basis_4 in range(fourier_basis+1):
						phi_s.append([basis_1, basis_2, basis_3, basis_4, basis_5])

	phi_s = np.array(phi_s)
	return phi_s

def main():

	plt.figure(figsize=(18, 16), dpi=100, facecolor='w', edgecolor='k')

	number_of_episodes = 100
	gamma = 1.0
	trials = 100

	epsilon = 1.0


	alpha1 = 1e-3
	alpha2 = 1e-3
	lambd = 0.8
	fourier_basis = 5
	greedy = False

	#for alpha1 in alphas_1:
	#	for alpha2 in alphas_2:
			#for epsilon in epsilons:
	#		for lambd in lambdas:
				#all_trial_data = []
	all_trial_data = np.zeros((trials, number_of_episodes))
	print("number of trials =", trials, "critic alpha =", alpha1, "actor alpha =", alpha2, "lambda =", lambd)
	for trial in range(trials):
		trial_data = evaluate(number_of_episodes, fourier_basis, gamma, alpha1, alpha2, epsilon, lambd, greedy)
		trial_data = np.array(trial_data).flatten()
		all_trial_data[trial] = trial_data


	#plt.plot(range(number_of_episodes), np.mean(all_trial_data, axis=0), color='blue', label="Softmax")
	plt.errorbar(range(len(all_trial_data[0])), np.mean(all_trial_data, axis=0), yerr=np.std(all_trial_data, axis=0), color='blue', ecolor='green', fmt='o')
	plt.xlabel('Episodes', fontsize=18)
	plt.ylabel('Expected Return', fontsize=18)
	#plt.legend()
	plt.show()

  
if __name__== "__main__":
  main()