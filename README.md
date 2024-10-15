# assignment_repository

1. Q-learning 및 Sarse Update 코드 작성

(1). Q-learning 
from collections import defaultdict
import numpy as np
class QLearning:
    def update(self, state, action, reward, next_state, next_action):
        state = self._convert_state(state)
        next_state = self._convert_state(next_state)

        q_value = self.q_values[state][action]

        ################## write code ################################
        # Get the maximum Q-value for the next state
        next_q_value = max(self.q_values[next_state])  
        
        # Compute the TD error
        td_error = reward + self.gamma * next_q_value - q_value
        
        # Update the Q-value for the current state and action
        self.q_values[state][action] += self.alpha * td_error
        ##############################################################

(2). SARAS
from collections import defaultdict
import numpy as np
class SARSA:
    def update(self, state, action, reward, next_state, next_action):
        state = self._convert_state(state)
        next_state = self._convert_state(next_state)
        
        q_value = self.q_values[state][action]

        ################ Write Code #####################
        # Get the Q-value for the next state-action pair
        next_q_value = self.q_values[next_state][next_action]  
        
        # Compute the TD error
        td_error = reward + self.gamma * next_q_value - q_value
        
        # Update the Q-value for the current state and action
        self.q_values[state][action] += self.alpha * td_error
        #################################################
