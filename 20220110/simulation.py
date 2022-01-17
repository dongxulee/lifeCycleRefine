from constant_simulation import *
from tqdm import tqdm
from jax import random

def simulation(beta_r, agentType, gamma, fileName):
    defineVariables(beta_r, agentType, gamma)
    #################################################################################################################### main function part 
    Vgrid = np.load(fileName + ".npy")
    df_1999 = pd.read_csv("df_1999_30to60.csv")
    
    if agentType == "richHigh":
        df = df_1999[(df_1999["skillLevel"] == "High")&(df_1999["financeExperience"] == "No")]
    elif agentType == "richLow":
        df = df_1999[(df_1999["skillLevel"] == "High")&(df_1999["financeExperience"] == "Yes")]
    elif agentType == "poorHigh":
        df = df_1999[(df_1999["skillLevel"] == "Low")&(df_1999["financeExperience"] == "No")]
    else:
        # agentType = "porrLow":
        df = df_1999[(df_1999["skillLevel"] == "Low")&(df_1999["financeExperience"] == "Yes")]
    
    df["ab"] = 30
    df["wealth"] = df["liquidWealth"] + df["investmentAmount"]
    codes = {'employed':1, 'unemployed': 0, "retired": 0}
    df["employmentStatus"] = df["employmentStatus"].map(codes)
    codes = {'owner':1, 'renter': 0}
    df["ownership"] = df["ownership"].map(codes)
    initialStates = df[["ageHead","wealth","ab","year","employmentStatus","ownership","participation"]]
    initialStates["year"] = imaginedEconState[0]
    initialStates = jnp.array(initialStates.values)

    # risk free interest rate depending on current S state 
    bondReturn = jnp.array(econRate[:,2])
    # stock return depending on current S state
    stockReturn = jnp.array(econRate[:,1])

    @partial(jit, static_argnums=(0,1))
    def transition_real(t,age,a,x):
        '''
            Input:
                x = [w,ab,s,e,o,z] single action 
                x = [0,1, 2,3,4,5] 
                a = [c,b,k,h,action] single state
                a = [0,1,2,3,4]
            Output:
                w_next
                ab_next
                s_next
                e_next
                o_next
                z_next

                prob_next
        '''
        s = jnp.array(x[2], dtype = jnp.int8)
        e = jnp.array(x[3], dtype = jnp.int8)
        # actions taken
        b = a[1]
        k = a[2]
        action = a[4]
        w_next = ((1+bondReturn[t])*b + (1+stockReturn[t])*k).repeat(nE)
        ab_next = (1-x[4])*(t*(action == 1)).repeat(nE) + x[4]*(x[1]*jnp.ones(nE))
        s_next = econ[t+1].repeat(nE)
        e_next = jnp.array([e,(1-e)])*(t+age-20<T_R) + jnp.array([0,0])*(t+age-20>=T_R)
        z_next = x[5]*jnp.ones(nE) + ((1-x[5]) * (k > 0)).repeat(nE)
        # job status changing probability and econ state transition probability
        pe = Pe[s, e]
        prob_next = jnp.array([1-pe, pe])
        # owner
        o_next_own = (x[4] - action).repeat(nE)
        # renter
        o_next_rent = action.repeat(nE)
        o_next = x[4] * o_next_own + (1-x[4]) * o_next_rent   
        return jnp.column_stack((w_next,ab_next,s_next,e_next,o_next,z_next,prob_next))

    '''
        # [w,ab,s,e,o,z]
        # w explicitly 
        # assume ab = 30 the strong assumption we made 
        # s is known 
        # e is known 
        # o is known
        # z is known
    '''
    def simulation(key, period = yearCount):
        x = initialStates[key.sum()%initialStates.shape[0]][1:]
        age = int(initialStates[key.sum()%initialStates.shape[0]][0])
        path = []
        move = []
        for t in range(0, period):
            key, subkey = random.split(key)
            if t+age-20 == T_max-1:
                _,a = V_solve(t+age-20,Vgrid[:,:,:,:,:,:,t+age-20],x)
            else:
                _,a = V_solve(t+age-20,Vgrid[:,:,:,:,:,:,t+age-20 + 1],x)
            xp = transition_real(t,age,a,x)           
            p = xp[:,-1]
            x_next = xp[:,:-1]
            path.append(x)
            move.append(a)
            x = x_next[random.choice(a = nE, p=p, key = subkey)]
        path.append(x)
        return jnp.array(path), jnp.array(move)

    # total number of agents
    num = initialStates.shape[0] * 10
    # simulation part 
    keys = vmap(random.PRNGKey)(jnp.arange(num))
    paths = []
    moves = []
    for i in tqdm(range(len(keys))):
        pp,mm = simulation(keys[i])
        paths.append(pp)
        moves.append(mm)
    Paths = jnp.array(paths)
    Moves = jnp.array(moves)

    # x = [w,ab,s,e,o,z]
    # x = [0,1, 2,3,4,5]
    _ws = Paths[:,:,0].T
    _ab = Paths[:,:,1].T
    _ss = Paths[:,:,2].T
    _es = Paths[:,:,3].T
    _os = Paths[:,:,4].T
    _zs = Paths[:,:,5].T
    _cs = Moves[:,:,0].T
    _bs = Moves[:,:,1].T
    _ks = Moves[:,:,2].T
    _hs = Moves[:,:,3].T
    _ms = Ms[jnp.append(jnp.array([0]),jnp.arange(yearCount)).reshape(-1,1) - jnp.array(_ab, dtype = jnp.int8)]*_os
    
    np.save("waseozcbkhm_" + fileName, np.array([_ws,_ab,_ss,_es,_os,_zs,_cs,_bs,_ks,_hs,_ms]))