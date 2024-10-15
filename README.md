# assignment_repository

1. Q-learning 및 Sarse Update 코드 작성

##(1). Q-learning 


        ################## write code ################################
        # Get the maximum Q-value for the next state
        next_q_value = max(self.q_values[next_state])  
        
        # Compute the TD error
        td_error = reward + self.gamma * next_q_value - q_value
        
        # Update the Q-value for the current state and action
        self.q_values[state][action] += self.alpha * td_error
        ##############################################################
        
###{s:np.round(q, 5).tolist() for s, q in agent.q_values.items()}

{0: [0.03591, 0.03582, 0.04015],
 3: [0.03229, 0.03238, 0.03694],
 12: [0.06365, 0.06374, 0.07088],
 24: [0.08437, 0.08473, 0.09556],
 27: [0.10944, 0.09882, 0.22622],
 39: [0.36317, 0.36092, 0.39958],
 15: [0.05046, 0.04499, 0.11092],
 6: [0.03414, 0.03389, 0.04091],
 36: [0.09784, 0.09766, 0.11886],
 30: [0.12324, 0.13746, 0.29273],
 33: [0.31158, 0.31751, 0.37407],
 21: [0.16025, 0.16062, 0.19216],
 18: [0.0586, 0.06441, 0.14],
 9: [0.04084, 0.0411, 0.04168],
 42: [0.48431, 0.48475, 0.52754],
 45: [0.0, 0.0, 0.0]}

 
##(2). SARAS

        ################ Write Code #####################
        # Get the Q-value for the next state-action pair
        next_q_value = self.q_values[next_state][next_action]  
        
        # Compute the TD error
        td_error = reward + self.gamma * next_q_value - q_value
        
        # Update the Q-value for the current state and action
        self.q_values[state][action] += self.alpha * td_error
        #################################################
        
###{s:np.round(q, 5).tolist() for s, q in agent.q_values.items()}

{0: [0.02611, 0.02623, 0.03145],
 12: [0.04818, 0.04822, 0.05413],
 24: [0.06594, 0.06624, 0.07717],
 36: [0.08409, 0.08452, 0.10545],
 3: [0.02495, 0.02503, 0.02615],
 6: [0.02887, 0.02913, 0.03277],
 18: [0.08428, 0.09228, 0.13304],
 27: [0.13606, 0.14523, 0.19461],
 15: [0.0645, 0.06231, 0.08614],
 30: [0.18967, 0.17792, 0.28841],
 9: [0.03142, 0.03169, 0.03774],
 21: [0.13561, 0.13588, 0.16969],
 39: [0.33982, 0.33997, 0.36706],
 33: [0.29578, 0.29552, 0.35773],
 45: [0.0, 0.0, 0.0],
 42: [0.46261, 0.46029, 0.51454]}
