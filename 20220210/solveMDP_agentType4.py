from solveMDP import *
from multiprocessing import Pool 

AgentType = ["richHigh"]
Beta_r = [0.10]
Gamma = [4.0]


def mapOverType(agentType):
    for beta_r in Beta_r:
         for gamma in Gamma:
            # discounting factor
            beta = 1/(1+beta_r)
            # utility function parameters 
            print("agentType: ", agentType)
            print("beta: ", beta)
            print("gamma: ", gamma)
            solveMDP(beta_r, agentType, gamma)

p = Pool(1)
p.map(mapOverType, AgentType)