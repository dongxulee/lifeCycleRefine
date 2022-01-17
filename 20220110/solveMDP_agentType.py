from solveMDP import *
from multiprocessing import Pool 

AgentType = ["poorHigh", "poorLow", "richHigh", "richLow"]
Beta_r = [0.01,0.03,0.05,0.07,0.09,0.10]
Gamma = [3.0, 4.0]


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

p = Pool(4)
p.map(mapOverType, AgentType)