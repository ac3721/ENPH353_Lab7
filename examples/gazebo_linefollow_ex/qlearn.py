import random
import pickle

 
class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''

        # Open the file containing the pickled data
        file = open(filename, 'rb')
        self.q = pickle.load(file)
        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        with open(filename, 'wb') as file:
            pickle.dump(self.q, file)
        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        maxQ = -500
        last = 0
        add = False
        action = 0
        ran = random.random()

        if (ran < self.epsilon):
            action = random.randint(0, 2)
        else:
            for act in range(3):
                #print(act, self.getQ(state, act))
                if (self.getQ(state, act) > maxQ) :
                    maxQ = self.getQ(state, act)
                    last = act
                elif (self.getQ(state, act) == 0):
                    if (last == 1 or act == 1):
                        add = True
                        action = 1
                    elif (act == 2 and last == 0):
                        add = True
                        action = 0
            if (add == False):
                action = last
                add = True
        # if (add == True):        
        #     print(action, "action", state)

        if (return_q == True):
            self.q[(state, action)] = self.getQ(state,action)
            return (action, self.q([state, action]))
        else:
            return action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        #self.q[(state1,action1)] = reward

        maxQ = -500
        for act in range(3):
            if (self.getQ(state2, act) > maxQ) :
                maxQ = self.getQ(state2, act)
                currentQ = self.getQ(state1, action1)
            if (self.getQ(state2, act) == 0):
                #self.q[(state1,action1)] = reward
                if (act == 1):
                    if (state2[0] == '1') :
                        maxQ = 3
                    if (state2[1] == '1') :
                        maxQ = 2
                    if (state2[2] == '1'):
                        maxQ = 2
                if (act == 0):
                    if (state2[3] == '1') :
                        maxQ = 2
                    if (state2[4] == '1') :
                        maxQ = 3
                    if (state2[5] == '1'):
                        maxQ = 3
                    if (state2[6] == '1'):
                        maxQ = 2
                if (act == 2):
                    if (state2[9] == 1) :
                        maxQ = 3
                    if (state2[8] == 1) :
                        maxQ = 2
                    if (state2[7] == 1):
                        maxQ = 2

        
        self.q[(state1,action1)] = self.getQ(state1, action1) + self.alpha * (reward + self.gamma * maxQ -self.getQ(state1, action1))
        #print(currentQ, "current", maxQ, "max",self.q[(state1,action1)])

    def savePolicy(self, filename):
        '''
        Save the epsilon, gamma, and alpha in a pickle file.
        '''
        data = [self.epsilon, self.gamma, self.alpha]

        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print("Wrote to file: {}".format(filename+".pickle"))

    
    def loadPolicy(self, filename):
        '''
        Load the epsilon, gamma, and alpha from a pickle file.

        '''
        file = open(filename, 'rb')
        self.epsilon, self.gamma, self.alpha = pickle.load(file)
        print("Loaded file: {}".format(filename+".pickle"))

