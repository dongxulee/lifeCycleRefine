from simulationCrossAge import *
import os.path
from multiprocessing import Pool 

AgentType = ["poorHigh", "poorLow", "richHigh", "richLow"]
Beta_r = [0.01,0.03,0.05,0.07,0.10]
Gamma = [3.0, 4.0]

def mapOverBeta(beta_r):
    for gamma in Gamma:
        for agentType in AgentType:
            fileName = agentType + "_" + str(beta_r) + "_" + str(gamma)
            if os.path.exists("crossAge_waseozcbkhm_" + fileName + ".npy"):
                break 
            if not os.path.exists(fileName + ".npy"):
                break
            '''
                Constants 
            '''
            # discounting factor
            beta = 1/(1+beta_r)
            # utility function parameter 
            print("agentType: ", agentType)
            print("beta: ", beta)
            print("gamma: ", gamma)
            simulation(beta_r, agentType, gamma, fileName)

p = Pool(5)
p.map(mapOverBeta, Beta_r)