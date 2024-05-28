from random import random
import numpy as np
import pandas as pd
from itertools import product
from math import sqrt
import matplotlib.pyplot as plt

def h(x):
    """ Convex function h(x) = x * log(x) with handling for x = 0. """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = x * np.log(x)
        result[np.isnan(result)] = 0  # Setting NaN results from 0 * log(0) to 0
    return result

def grad_h(x):
    """ Gradient of h(x) = 1 + log(x) with handling for x = 0. """
    with np.errstate(divide='ignore'):
        result = 1 + np.log(x)
        result[np.isneginf(result)] = 0  # Handling -inf results from log(0)
    return result


def bregman_divergence(p, q, h, grad_h):
    """Calculate the Bregman divergence from q to p using function h."""
    h_p = h(p)
    h_q = h(q)
    grad_h_q = grad_h(q)
    divergence = h_p - h_q - np.dot(grad_h_q, (p - q))
    return divergence



class Det_Game:    
    def __init__(self, action_list, cost_list):
        self.cost_list = cost_list
        self.agent_actions = self.initialize_actions(action_list)
        self.agent_weights = self.initialize_weights(self.agent_actions)
        All_Strategy = list(product(*[i for i in action_list]))
        self.action_to_cost = self.initialize_actions_cost(All_Strategy, cost_list)
        self.nash_equilibrium = self.calculate_ne()

    def calculate_ne(self, default_mixed_strategy=0.5):
        # The cost_list should reflect the payoff structure you've given:
        # cost_list = [[(1+delta, -1-delta), (-1, 1)],
        #              [(-1, 1), (1, -1)]]
        # This is structured as:
        # [[Payoff when both choose Head, Payoff when P1 chooses Head and P2 chooses Tail],
        #  [Payoff when P1 chooses Tail and P2 chooses Head, Payoff when both choose Tail]]
        
        # Map payoffs to a more accessible dictionary
        payoffs = {
            ('Head', 'Head'): self.cost_list[0][0],
            ('Head', 'Tail'): self.cost_list[0][1],
            ('Tail', 'Head'): self.cost_list[1][0],
            ('Tail', 'Tail'): self.cost_list[1][1]
        }

        # Extracting payoffs for Player 1 (P1)
        a, b = payoffs[('Head', 'Head')][0], payoffs[('Head', 'Tail')][0]
        c, d = payoffs[('Tail', 'Head')][0], payoffs[('Tail', 'Tail')][0]

        # Extracting payoffs for Player 2 (P2)
        e, f = payoffs[('Head', 'Head')][1], payoffs[('Head', 'Tail')][1]
        g, h = payoffs[('Tail', 'Head')][1], payoffs[('Tail', 'Tail')][1]

        # Calculate equilibrium probabilities for Player 1
        # p is P1's probability of choosing 'Head'
        denom_p = a - b - c + d
        p = (d - c) / denom_p if denom_p != 0 else default_mixed_strategy

        # Calculate equilibrium probabilities for Player 2
        # q is P2's probability of choosing 'Head'
        denom_q = e - f - g + h
        q = (h - f) / denom_q if denom_q != 0 else default_mixed_strategy

        # Ensure probabilities are within [0, 1]
        p, q = max(0, min(p, 1)), max(0, min(q, 1))
        return {'Player 1': {'Head': p, 'Tail': 1 - p}, 'Player 2': {'Head': q, 'Tail': 1 - q}}
    

    
    def calculate_bregman_divergence(self):
        history1 = np.array(self.agent1_history)[:, 0]  # Probabilities of choosing 'Head' for agent1
        history2 = np.array(self.agent2_history)[:, 0]  # Probabilities of choosing 'Head' for agent2

        # Nash Equilibrium strategies
        ne_p1_head = self.nash_equilibrium['Player 1']['Head']
        ne_p2_tail = self.nash_equilibrium['Player 2']['Tail']

        # Compute Bregman divergence for each agent's strategy history
        divergences1 = [bregman_divergence(np.array([ne_p1_head, 1-ne_p1_head]), np.array([p, 1-p]), h, grad_h) for p in history1]
        divergences2 = [bregman_divergence(np.array([ne_p2_tail, 1-ne_p2_tail]), np.array([p, 1-p]), h, grad_h) for p in history2]

        return divergences1, divergences2

    def calculate_statistics(self):
        # First, calculate the divergences if not already calculated
        divergences1, divergences2 = self.calculate_bregman_divergence()

        # Convert list of arrays into a single array for easier manipulation
        divergences1 = np.array(divergences1)
        divergences2 = np.array(divergences2)

        # Since it's a zero-sum game, you might want to analyze the absolute divergence for one player only
        # as the results are symmetric
        abs_divergences = np.abs(divergences1)  # Using Player 1 as representative

        # Compute statistical measures
        mean_divergence = np.mean(abs_divergences)
        max_divergence = np.max(abs_divergences)
        min_divergence = np.min(abs_divergences)
        variance_divergence = np.var(abs_divergences)
        std_deviation = np.std(abs_divergences)

        # Create a dictionary to store these statistics
        stats = {
            'Mean Divergence': mean_divergence,
            'Max Divergence': max_divergence,
            'Min Divergence': min_divergence,
            'Variance of Divergence': variance_divergence,
            'Standard Deviation of Divergence': std_deviation
        }

        return stats


    def initialize_weights(self,agent_actions_dict):
        agent_weights = {}
        for agent_name,strategy in agent_actions_dict.items():
            agent_weights[agent_name] = np.ones(len(strategy)) +np.array([0.2,0.1])#+ np.random.normal(0,0.1, size=len(strategy))
            # Cannot start with 1, 1 ,1 weight, otherwise it will get stuck in the MNE already
#             agent_weights[agent_name] = np.array([1,2])
        return agent_weights

    
    def initialize_actions(self,agent_actions_list):
        agent_actions = {}
        agent_counter = 1
        for actions in agent_actions_list:
            agent_actions['agent{}'.format(agent_counter)] = actions
            agent_counter +=1
        return agent_actions        

    def initialize_actions_cost(self,All_Strategy, cost_list):
        """cost_list should be input row by row"""
        action_to_cost = {}
        flat_cost_list = [item for sublist in cost_list for item in sublist]
        for strategy, cost in zip(All_Strategy,flat_cost_list):
            action_to_cost[strategy] = cost
        return action_to_cost

    def action(self,probability_distribution):
        return np.random.choice(len(probability_distribution),1,p=probability_distribution)[0] # [0] unwarp the array

    def get_action(self,agent, agent_weights):
        total = sum(agent_weights[agent])
        num_actions = len(agent_weights[agent])
        agent_p = [agent_weights[agent][i]/total for i in range(num_actions)]
        debug_agent_p1 = agent_weights[agent][1]/total
        agent_a = self.action(agent_p)
    #     print('probability of action1 = {}'.format(agent_p))
    #     print('probability of action2 = {}'.format(debug_agent_p1))
    #     print(agent_weights[agent])

        return agent_a, agent_p  

    def get_cost_vector(self, agent, opponent_p, agent_actions, action_to_cost):
#         print("debug agent_actions = ", agent_actions)
        if agent == 'agent1':
            num_actions = len(agent_actions[agent])
            cost_vector = []
            action_vector = [[tuple([agent_actions['agent1'][i],agent_actions['agent2'][j]]) for j in range(num_actions)] for i in range(num_actions)]
            # Now assume 2 agents have the same number of strategy
            action_vector = np.array([list(map(action_to_cost.get, a)) for a in action_vector])[:,:,0]
            for a in action_vector:
                cost_vector.append(np.sum(a*opponent_p)) # 0 means agent1
        else:
            num_actions = len(agent_actions[agent])
            action_vector = [[tuple([agent_actions['agent1'][j],agent_actions['agent2'][i] ]) for j in range(num_actions)] for i in range(num_actions)]
            action_vector = np.array([list(map(action_to_cost.get, a)) for a in action_vector])[:,:,1]
            cost_vector = []
            for a in action_vector:
                cost_vector.append(np.sum(a*opponent_p))
        return np.array(cost_vector)

    def print_game(self):
        pass
        
    def MWU(self,iterations, epsilon=0.05, new_ep=lambda ep, t:ep): 
        # Assuming you have the NE probabilities calculated and want to compare them:
        print("Starting MWU with Nash Equilibrium:")
        print(self.nash_equilibrium)
        epsilon_history = []
        num_actions1 = len(self.agent_weights['agent1'])
        num_actions2 = len(self.agent_weights['agent2'])
        
        agent1_sum = sum(self.agent_weights['agent1'])
        agent2_sum = sum(self.agent_weights['agent2'])

        self.agent1_history = [[self.agent_weights['agent1'][i]/agent1_sum for i in range(num_actions1)]]
        self.agent2_history = [[self.agent_weights['agent2'][i]/agent2_sum for i in range(num_actions2)]]

        counter = 1
        for i in range(iterations):
    #         print('playing game round {}'.format(counter))
            agent1_a, agent1_p = self.get_action('agent1',self.agent_weights)
            agent2_a, agent2_p = self.get_action('agent2',self.agent_weights)
#             print("weight vector = ",self.agent_weights['agent1'])
            cost_vector1 = self.get_cost_vector('agent1', agent2_p, self.agent_actions, self.action_to_cost)
            cost_vector2 = self.get_cost_vector('agent2', agent1_p, self.agent_actions, self.action_to_cost)
            epsilon = new_ep(epsilon, i+1)
            epsilon_history.append(epsilon)
            self.agent_weights['agent1'] = self.agent_weights['agent1']*(1-epsilon)**cost_vector1
            self.agent_weights['agent2'] = self.agent_weights['agent2']*(1-epsilon)**cost_vector2    

            agent1_sum = sum(self.agent_weights['agent1'])
            agent2_sum = sum(self.agent_weights['agent2'])

            self.agent1_history.append([self.agent_weights['agent1'][i]/agent1_sum for i in range(num_actions1)])
            self.agent2_history.append([self.agent_weights['agent2'][i]/agent2_sum for i in range(num_actions2)])


            counter+=1
        print("Training for {} steps is done".format(counter-1))
        return epsilon_history
#         print('outputing {} strategies for agent1'.format(num_actions1))
#         print('outputing {} strategies for agent2'.format(num_actions2))
#         return self.agent1_history, self.agent2_history
    
    def plot(self):
        fontsize = 16
        fontsize_title = 20
        fig, ax = plt.subplots(1,2,figsize=(16,5))
        ax[0].set_title('agent1', fontsize=fontsize_title)
        ax[1].set_title('agent2', fontsize=fontsize_title)
        ax[0].set_xlabel('step', fontsize=fontsize)
        ax[1].set_xlabel('step', fontsize=fontsize)
        ax[0].set_ylabel('Probability of choosing certain action', fontsize=fontsize)
        ax[1].set_ylabel('Probability of choosing certain action', fontsize=fontsize)    

        ax[0].plot(self.agent1_history)
        ax[1].plot(self.agent2_history)

        ax[0].legend([s for s in self.agent_actions['agent1']])
        ax[1].legend([s for s in self.agent_actions['agent2']])
        
    def show_game(self):
        return pd.DataFrame(self.cost_list,columns=self.agent_actions['agent2'], index=self.agent_actions['agent1'])

    def get_history(self):
#         print(len(self.agent1_history))
#         print(len(self.agent2_history))
        return [self.agent1_history, self.agent2_history]
    
    def plot_scatter(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(np.array(self.agent1_history)[:, 0], np.array(self.agent2_history)[:, 0], color='black')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Dynamically mark the Nash Equilibrium point
        ne_p1_head = self.nash_equilibrium['Player 1']['Head']
        ne_p2_tail = self.nash_equilibrium['Player 2']['Tail']
        plt.scatter(ne_p1_head, ne_p2_tail, color='blue', s=100, zorder=5)

        # Calculate dynamic offset based on the y-axis range
        y_range = plt.ylim()  # Get the current y-axis limits
        offset = (y_range[1] - y_range[0]) * 0.02  # Dynamic offset, e.g., 2% of the y-axis range

        plt.text(ne_p1_head, ne_p2_tail + offset, '$x^{NE}$', ha='center', va='center', fontsize=14, color='blue')

        # Label corners
        plt.text(0, 1.05, 'Tails, Heads', ha='center', va='center', fontsize=14)
        plt.text(0, -0.05, 'Tails, Tails', ha='center', va='center', fontsize=14)
        plt.text(1, 1.05, 'Heads, Heads', ha='center', va='center', fontsize=14)
        plt.text(1, -0.05, 'Heads, Tails', ha='center', va='center', fontsize=14)

        # Set the aspect of the plot to be equal
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()