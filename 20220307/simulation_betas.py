from simulation import *
import os.path
from multiprocessing import Pool 
import warnings
warnings.filterwarnings("ignore")

AgentType = ["poorHigh", "poorLow", "richHigh", "richLow"]
Beta_r = [0.06]
Gamma = [4.0]

def mapOverBeta(beta_r):
    for gamma in Gamma:
        for agentType in AgentType:
            fileName = agentType + "_" + str(beta_r) + "_" + str(gamma)
            if os.path.exists("waseozcbkhm_" + fileName + ".npy"):
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

p = Pool(1)
p.map(mapOverBeta, Beta_r)