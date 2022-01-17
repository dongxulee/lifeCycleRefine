from constant import *
from tqdm import tqdm
import os.path

def solveMDP(beta_r, agentType, gamma):
    # define varilables based on beta_r, agentType, gamma
    defineVariables(beta_r, agentType, gamma)
    ###################################solving the model################################################## 
    fileName = agentType + "_" + str(beta_r) + "_" + str(gamma)
    if os.path.exists(fileName + ".npy"):
        print("Model Solved! ")
    else:
        for t in tqdm(range(T_max-1,T_min-1, -1)):
            if t == T_max-1:
                v = vmap(partial(V,t,Vgrid[:,:,:,:,:,:,t]))(Xs)
            else:
                v = vmap(partial(V,t,Vgrid[:,:,:,:,:,:,t+1]))(Xs)
            Vgrid[:,:,:,:,:,:,t] = v.reshape(dim)
        np.save(fileName,Vgrid)