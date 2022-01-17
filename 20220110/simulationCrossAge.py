from constant import * 
from jax import random

def simulation(beta_r, agentType, gamma, fileName):
    defineVariables(beta_r, agentType, gamma)
    ##################################################################################### main function part 
    # total number of agents
    num = 10000
    '''
        x = [w,ab,s,e,o,z]
        x = [5,0, 0,0,0,0]
    '''
    from quantecon import MarkovChain
    # number of economies and each economy has 100 agents
    numEcon = 100
    numAgents = num//100
    mc = MarkovChain(Ps)
    econStates = mc.simulate(ts_length=T_max-T_min,init=0,num_reps=numEcon)
    econStates = jnp.array(econStates,dtype = int)
    
    @partial(jit, static_argnums=(0,))
    def transition_real(t,a,x, s_prime):
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
        w_next = ((1+r_b[s])*b + (1+r_k[s_prime])*k).repeat(nE)
        ab_next = (1-x[4])*(t*(action == 1)).repeat(nE) + x[4]*(x[1]*jnp.ones(nE))
        s_next = s_prime.repeat(nE)
        e_next = jnp.array([e,(1-e)])
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


    def simulation(key):
        initE = random.choice(a = nE, p=E_distribution, key = key)
        initS = random.choice(a = nS, p=S_distribution, key = key) 
        x = [0, 0, initS, initE, 0, 0]
        path = []
        move = []
        # first 100 agents are in the 1st economy and second 100 agents are in the 2nd economy 
        econ = econStates[key.sum()//numAgents,:]
        for t in range(T_min, T_max):
            _, key = random.split(key)
            if t == T_max-1:
                _,a = V_solve(t,Vgrid[:,:,:,:,:,:,t],x)
            else:
                _,a = V_solve(t,Vgrid[:,:,:,:,:,:,t+1],x)
            xp = transition_real(t,a,x, econ[t])            
            p = xp[:,-1]
            x_next = xp[:,:-1]
            path.append(x)
            move.append(a)
            x = x_next[random.choice(a = nE, p=p, key = key)]
        path.append(x)
        return jnp.array(path), jnp.array(move)

    # simulation part 
    keys = vmap(random.PRNGKey)(jnp.arange(num))
    Paths, Moves = vmap(simulation)(keys)
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
    _ms = Ms[jnp.append(jnp.array([0]),jnp.arange(T_max)).reshape(-1,1) - jnp.array(_ab, dtype = jnp.int8)]*_os
    
    np.save("crossAge_waseozcbkhm_" + fileName, np.array([_ws,_ab,_ss,_es,_os,_zs,_cs,_bs,_ks,_hs,_ms]))