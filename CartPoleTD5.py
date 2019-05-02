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
		self.norm_state[0] = (self.state[0] + 3.0) / 6.0
		self.norm_state[1] = (self.state[1] + 10.0) / 20.0
		self.norm_state[2] = (self.state[2] + self.fail_angle) / (self.fail_angle * 2.0)
		self.norm_state[3] = (self.state[3] + np.pi) / (np.pi * 2.0)


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
	def __init__(self, action_space, policy, gamma = 1.0):
		self.policy = policy
		self.action_space = action_space
		self.sumOfRewards = 0
		self.gamma = 1.0
		#self.final_state = final_state

	def make_random_move(self):
		move = np.random.choice(a=[-1, 1], size=1, p=[0.5, 0.5])
		return move

	def response(self, reward, time):
		self.sumOfRewards += (self.gamma ** time) * reward
		return
		
	def give_policy(self):
		return self.policy

	def run(self, MDP, state_values, gamma, alpha, train, phi_s, weight_matrix):
		td_error = []
		position = []
		state = MDP.state
		time_step = 0.0
		while time_step != MDP.max_time:
			previous_position = position.copy()
			if train:
				position, reward, time_step = MDP.change_state(self.make_random_move())
				if previous_position:
					weight_matrix = TD_update(previous_position, position, reward, gamma, alpha, phi_s, weight_matrix)
			else:
				position, reward, time_step = MDP.change_state(self.make_random_move())
				if previous_position:
					state_phi_s = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(phi_s, previous_position)))
					new_state_phi_s = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(phi_s, position)))
					td_error.append((reward + gamma*new_state_phi_s - state_phi_s)**2)

			if reward == MDP.fail_reward:
				break
		return state_values, td_error, weight_matrix



def evaluate(state_values, number_of_episodes, phi_s, weight_matrix, gamma, alpha, train):
	all_td_errors = []
	for episode in range(number_of_episodes):
		MDP = CartPoleMDP()
		state_space, action_space = MDP.give_MDP_info()
		episode_agent = Agent(action_space, state_values, gamma)
		state_values, td_errors, weight_matrix = episode_agent.run(MDP, state_values, gamma, alpha, train, phi_s, weight_matrix)
		if not train:
			all_td_errors.extend(td_errors)
	all_td_errors = np.array(all_td_errors)

	return state_values, all_td_errors, weight_matrix

def TD_update(state, new_state, reward, gamma, alpha, phi_s, weight_matrix):
	state_phi_s = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(phi_s, state)))
	new_state_phi_s = np.matmul(weight_matrix.T, np.cos(np.pi * np.dot(phi_s, new_state)))

	error = reward + gamma*new_state_phi_s - state_phi_s
	weight = weight_matrix.T + (error*alpha*np.cos(np.pi * np.dot(phi_s, state))).T
	return weight.T


def main():

	plt.figure(figsize=(18, 16), dpi=100, facecolor='w', edgecolor='k')

	number_of_episodes = 100
	gamma = 1.0

	phi_s = []
	for basis_1 in range(6):
		for basis_2 in range(6):
			for basis_3 in range(6):
				for basis_4 in range(6):
					phi_s.append([basis_1, basis_2, basis_3, basis_4])


	phi_s = np.array(phi_s)

	#alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
	alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
	plot_alphas = np.log10(alphas)

	ind = np.arange(len(alphas))

	MSE_td = []

	print("number of episodes:", number_of_episodes)

	for alpha in alphas:
		state_values = [0.0] * 14641 # 11 possible values for each of the 4 states 11*11*11*11
		weight_matrix = np.zeros(len(phi_s))
		state_values, _, weight_matrix= evaluate(state_values, number_of_episodes, phi_s, weight_matrix, gamma=gamma, alpha=alpha, train = True)
		state_values, td_errors, _ = evaluate(state_values, number_of_episodes, phi_s, weight_matrix, gamma=gamma, alpha=alpha, train = False)
		MSE_td.append(np.mean(td_errors))
		print(alpha, len(td_errors), np.mean(td_errors))


	plt.plot(plot_alphas, MSE_td, label="CartPole5")
	#plt.yscale("log")

	#plt.bar(np.arange(len(MSE_td)), MSE_td, width=0.1)
	#plt.xticks(ind, alphas)

	plt.xlabel(r"$log_{10}{(Step Size)}$", fontsize=18)
	plt.ylabel('Mean Squared TD Error', fontsize=18)
	plt.legend()

	plt.show()
  
if __name__== "__main__":
  main()