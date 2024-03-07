import random
import pickle

 
class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions
        self.past = []

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

        maxQ = 0
        last = 0
        add = False
        action = 0
        ran = random.random()

        if (ran < self.epsilon):
            action = random.randint(0, 3)
            print(ran, "ran", action)

        else:
            for act in range(3):
                print(act, self.getQ(state, act))
                if (self.getQ(state, act) > maxQ) :
                    maxQ = self.getQ(state, act)
                    last = act
                elif (self.getQ(state, act) == maxQ):
                    if (last == 0):
                        add = True
                        action = 0
                    elif (act == 2 and last == 1):
                        add = True
                        action = 1
            if (add == False):
                action = last

        self.past.append(action)

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
        self.q[(state1,action1)] = reward

        currentQ = self.getQ(state1, action1)

        maxQ = 0
        for act in range(3):
            if (self.getQ(state2, act) > maxQ) :
                maxQ = self.getQ(state2, act)


        self.q[(state1,action1)] += self.alpha * (reward + self.gamma * maxQ -currentQ)