import numpy as np
from numpy.random import normal
from numpy.random import uniform
import matplotlib.pyplot as plt

class CartPoleMDP:
	def __init__(self):
		self.fail_angle = np.pi/2
		self.motor_force = 10.0
		self.gravity = 9.8
		self.m_c = 1.0
		self.m_p = 0.1
		self.l_p = 0.5
		self.force = 1.0
		self.time_step = 0.02
		self.time = 0.0
		self.max_time = 20.0 + 10.0*self.time_step
		self.reward = 1.0
		self.final_reward = 1.0
		self.fail_reward = -10.0

		self.boundary = [-3, 3]
		self.state = [0.0, 0.0, 0.0, 0.0] # position, velocity, angle, angular velocity
		self.norm_state = [0.0, 0.0, 0.0, 0.0]
		self.actions = [-1, 1]
		self.gamma = 1.0
		self.states = 4

	def give_MDP_info(self):
		return self.states, len(self.actions)#, self.final_state

	def change_state(self, action):
		self.time = round(self.time + self.time_step, 2)

		force = action * self.motor_force
		ang_acc = self.angular_acceleration(force)
		acc = self.acceleration(ang_acc, force)

		self.state[1] = self.forward_euler_method(self.state[1], acc)
		self.state[0] = self.forward_euler_method(self.state[0], self.state[1])

		self.state[3] = self.forward_euler_method(self.state[3], ang_acc)
		self.state[2] = self.forward_euler_method(self.state[2], self.state[3])

		# normalize the state
		self.norm_state[0] = (2.0 * (self.state[0] + 3.0) / 6.0) - 1.0
		self.norm_state[1] = (2.0 * (self.state[1] + 10.0) / 20.0) - 1.0
		self.norm_state[2] = (2.0 * (self.state[2] + self.fail_angle) / (self.fail_angle * 2.0)) - 1.0
		self.norm_state[3] = (2.0 * (self.state[3] + np.pi) / (np.pi * 2.0)) - 1.0


		if self.state[1] < -10.0 or self.state[1] > 10.0:
			return self.norm_state, self.fail_reward, self.time

		if self.state[3] < -np.pi or self.state[3] > np.pi:
			return self.norm_state, self.fail_reward, self.time

		if self.state[0] > 3.0 or self.state[0] < -3.0:
			return self.norm_state, self.fail_reward, self.time

		if self.state[2] < -self.fail_angle or self.state[2] > self.fail_angle:
			return self.norm_state, self.fail_reward, self.time

		if(self.time == self.max_time):
			return self.norm_state, self.final_reward, self.time

		return self.norm_state, self.reward, self.time

	def angular_acceleration(self, force):
		numerator = self.gravity*np.sin(self.state[2]) + np.cos(self.state[2]) * ((-force - self.m_p*self.l_p*self.state[3]*self.state[3]*np.sin(self.state[2]))/(self.m_p + self.m_c))
		denominator = self.l_p * (4.0/3.0 - (self.m_p*np.cos(self.state[2])*np.cos(self.state[2]))/(self.m_c + self.m_p))
		return numerator/denominator

	def acceleration(self, angular_acceleration, force):
		numerator = force + self.m_p*self.l_p*(self.state[3]*self.state[3]*np.sin(self.state[2])-angular_acceleration*np.cos(self.state[2]))
		return (numerator/(self.m_c+self.m_p))

	def forward_euler_method(self, y, y_dot):
		return (y + self.time_step*y_dot)


class Agent:
	def __init__(self, action_space, gamma, q_table, epsilon, alpha):
		self.action_space = action_space
		self.sumOfRewards = 0.0
		self.gamma = gamma
		self.q_table = q_table
		self.epsilon = epsilon
		self.alpha = alpha

	def make_random_move(self):
		move = np.random.choice(a=[-1, 1], size=1, p=[0.5, 0.5])
		return move

	def make_policy_move(self, state, weight_matrix):
		left_state = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(self.q_table, (state + [1, 0]))))
		right_state = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(self.q_table, (state + [0, 1]))))

		if left_state > right_state:
			action = 0
			action_matrix = [1, 0]
		else:
			action = 1
			action_matrix = [0, 1]

		prob_array = [self.epsilon/2.0, self.epsilon/2.0]
		prob_array[action] = 1.0 - self.epsilon + (self.epsilon/2.0)
		move = np.random.choice(a=[-1, 1], size=1, p=prob_array)[0]
		return move, action_matrix

	def make_softmax_move(self, state, weight_matrix):
		left_state = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(self.q_table, (state + [1, 0]))))
		right_state = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(self.q_table, (state + [0, 1]))))

		table = np.array([left_state, right_state])

		prob_array = np.exp(self.epsilon * table - np.max(self.epsilon * table)) / np.sum(np.exp(self.epsilon*table - np.max(self.epsilon * table)))
		#print(prob_array)
		move = np.random.choice(a=[-1, 1], size=1, p=prob_array)[0]
		if move == -1:
			action_matrix = [1, 0]
		else:
			action_matrix = [0, 1]
		return move, action_matrix

	def response(self, reward):
		self.sumOfRewards += reward
		return
		
	def give_policy(self):
		return self.policy

	def run(self, MDP, weight_matrix):
		position = [0.0, 0.0, 0.0, 0.0]
		#position = [0.0, 0.0, 0.0, 0.0]
		action, action_matrix = self.make_policy_move(position, weight_matrix)
		time_step = 0.0
		while time_step != MDP.max_time:
			next_position, reward, next_time_step = MDP.change_state(action)
			self.response(reward)
			next_action, next_action_matrix = self.make_policy_move(next_position, weight_matrix)
			weight_matrix = TD_update(position, action_matrix, next_position, next_action_matrix, reward, self.gamma, self.alpha, self.q_table, weight_matrix)
			position = next_position.copy()
			action_matrix = next_action_matrix.copy()
			action = next_action
			time_step = next_time_step

			if reward == MDP.fail_reward:
				break
		return weight_matrix, self.sumOfRewards



def evaluate(number_of_episodes, polynomial_basis, gamma, alpha, epsilon):
	q_table = make_CP_phi(polynomial_basis)
	expected_returns = []
	weight_matrix = np.zeros(len(q_table))
	count = 0
	for episode in range(number_of_episodes):
		MDP = CartPoleMDP()
		state_space, action_space = MDP.give_MDP_info()
		episode_agent = Agent(action_space, gamma, q_table, epsilon, alpha)
		weight_matrix, expected_return = episode_agent.run(MDP, weight_matrix)
		expected_returns.append(expected_return)
		if expected_return >= 50.0:
			count+=1
		#epsilon *= 0.9
		#alpha *= 0.7
	#print(count, max(expected_returns), np.mean(expected_returns))
	#print(expected_returns)
		
	return expected_returns

#def TD_update(state, new_state, reward, gamma, alpha, phi_s, weight_matrix):
#	state_phi_s = np.matmul(weight_matrix.T, calc_poly_basis(phi_s, state))
#	new_state_phi_s = np.matmul(weight_matrix.T, calc_poly_basis(phi_s, new_state))

#	error = reward + gamma*new_state_phi_s - state_phi_s
#	weight = weight_matrix.T + (error*alpha*calc_poly_basis(phi_s, state)).T
#	return weight.T

def TD_update(state, action, new_state, new_action, reward, gamma, alpha, phi_s, weight_matrix):
	state_phi_s = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(phi_s, state + action)))
	new_state_phi_s = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(phi_s, new_state + new_action)))

	error = reward + gamma*new_state_phi_s - state_phi_s

	weight = weight_matrix.T + (error*alpha*np.cos(np.pi * np.dot(phi_s, state + action))).T
	return weight.T

def calc_poly_basis(basis_vector, state):
	return np.prod(state**basis_vector, axis=1)


def make_CP_phi(polynomial_basis):
	phi_s = []
	for basis_1 in range(polynomial_basis):
		for basis_2 in range(polynomial_basis):
			for basis_3 in range(polynomial_basis):
				for basis_4 in range(polynomial_basis):
					for basis_5 in range(polynomial_basis):
						for basis_6 in range(polynomial_basis):
							phi_s.append([basis_1, basis_2, basis_3, basis_4, basis_5, basis_6])

	phi_s = np.array(phi_s)
	return phi_s


def main():

	plt.figure(figsize=(18, 16), dpi=100, facecolor='w', edgecolor='k')

	number_of_episodes = 100
	gamma = 1.0
	polynomial_basis = 4
	trials = 100
	#alpha = 0.0001

	#alphas = np.random.uniform(0.000001, 0.0001, 5)
	#epsilons = np.random.uniform(0.0001, 0.1, 5)


	#epsilon = 0.01814120335386342 alpha = 6.811968249039493e-05
	epsilon = 0.01158487412671914 
	alpha = 0.0005760185539576551

	# graphed values
	#epsilon = 0.050564293396948845 
	#alpha = 8.125435228711253e-05

	#for alpha in alphas:
	#	for epsilon in epsilons:
	all_trial_data = np.zeros((trials, number_of_episodes))
	print("number of episodes =", number_of_episodes, "number of trials =", trials, "epsilon =", epsilon, "alpha =", alpha)
	for trial in range(trials):
		trial_data = evaluate(number_of_episodes, polynomial_basis, gamma, alpha, epsilon)
		trial_data = np.array(trial_data).flatten()
		all_trial_data[trial] = trial_data


	plt.errorbar(range(len(all_trial_data[0])), np.mean(all_trial_data, axis=0), yerr=np.std(all_trial_data, axis=0), ecolor='yellow', fmt='o')
	plt.xlabel('Episodes', fontsize=18)
	plt.ylabel('Expected Return', fontsize=18)
	plt.show()
			#plt.gcf().clear()
			#plt.figure(figsize=(18, 16), dpi=100, facecolor='w', edgecolor='k')
  
if __name__== "__main__":
  main()