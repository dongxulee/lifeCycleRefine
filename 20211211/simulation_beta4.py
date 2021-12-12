import numpy as np
import jax.numpy as jnp
from jax.numpy import interp
from jax import jit, partial, random, vmap
from tqdm import tqdm
import pandas as pd
import warnings
import os.path
warnings.filterwarnings("ignore")
np.printoptions(precision=2)

AgentType = ["poorHigh","poorLow","richHigh","richLow"]
Beta_r = [0.04]
Gamma = [3.0]

for ga in Gamma:
    for beta_r in Beta_r:
        for agentType in AgentType:
            gamma = ga
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
            
            # time line, starts at 20 ends at 80
            T_min = 0
            T_max = 60
            T_R = 45
            # relative importance of housing consumption and non durable consumption 
            alpha = 0.7
            # parameter used to calculate the housing consumption 
            kappa = 0.3
            # uB associated parameter
            B = 2.0
            # social welfare after the unemployment
            welfare = 20
            # tax rate before and after retirement
            tau_L = 0.2
            tau_R = 0.1
            # number of states S
            nS = 8
            # number of states e
            nE = 2
            # housing state
            nO = 2
            # experience state 
            nZ = 2


            '''
                Economic state calibration 
            '''

            # probability of survival
            Pa = jnp.array(np.load("constant/prob.npy"))
            ############################################################################################################ skill and finance literacy 
            if "rich" in agentType:
                # deterministic income
                detEarning = jnp.array(np.load("constant/highIncomeDetermined.npy"))
            else:
                detEarning = jnp.array(np.load("constant/lowIncomeDetermined.npy"))

            if "High" in agentType:
                # stock transaction fee
                Kc = 0.02
                # stock participation cost
                c_k = 20
            else:
                # stock transaction fee
                Kc = 0
                # stock participation cost
                c_k = 5
            ############################################################################################################
            # Define transition matrix of economical states S
            Ps = np.genfromtxt('constant/Ps.csv',delimiter=',')
            Ps = jnp.array(Ps)
            # The possible GDP growth, stock return, bond return
            gkfe = np.genfromtxt('constant/gkfe.csv',delimiter=',')
            gkfe = jnp.array(gkfe)
            # GDP growth depending on current S state
            gGDP = gkfe[:,0]/100
            # risk free interest rate depending on current S state 
            r_b = gkfe[:,1]/100
            # stock return depending on current S state
            r_k = gkfe[:,2]/100
            # unemployment rate depending on current S state 
            Pe = gkfe[:,7:]/100
            Pe = Pe[:,::-1]


            '''
                Real Econ Shock calibration
            '''

            # empirical econ
            empiricalEcon = pd.read_csv('constant/empiricalEcon.csv',delimiter=',')
            empiricalEcon = empiricalEcon.set_index("year")
            empiricalEcon = empiricalEcon/100
            # match the empirical states in memoryState
            memoryState = np.column_stack((gGDP, r_k, r_b))
            def similarity(actualState, memoryState = memoryState):
                '''
                    state is charactorized as 3 dim vector
                '''
                diffState = np.sum(np.abs(actualState - memoryState), axis = 1)
                distance = np.min(diffState)
                state = np.argmin(diffState)
                return distance, state
            similarity, imaginedEconState = np.vectorize(similarity, signature='(n)->(),()')(empiricalEcon.values)
            # generate economic states of a certain time window
            def generateEcon(yearBegin, yearCount,imaginedEconState,empiricalEcon):
                # single economy generation
                years = empiricalEcon.index.values
                econ = jnp.array(imaginedEconState[np.where(years == yearBegin)[0][0]:np.where(years == yearBegin)[0][0]+yearCount],dtype = int)
                econRate = empiricalEcon[np.where(years == yearBegin)[0][0]:np.where(years == yearBegin)[0][0]+yearCount].values
                return econ, econRate
            #**********************************simulation change*****************************************************#
            yearBegin = 1999
            yearCount = 20
            econ, econRate = generateEcon(yearBegin, yearCount,imaginedEconState,empiricalEcon)



            '''
                calculate stationary distribution to prepare for simulation
            '''
            # calculate the stationary distribution of econ state and employment state
            S_distribution = jnp.ones(nS)/nS
            for _ in range(100):
                S_distribution = jnp.matmul(S_distribution, Ps)

            #P(0,1)
            P01 = jnp.dot(Pe[:,0],S_distribution)
            #P(1,0)
            P10 = jnp.dot(Pe[:,1],S_distribution)
            jnp.array([[1-P01, P01],[P10, 1-P10]])

            E_distribution = jnp.ones(2)/2
            for _ in range(100):
                E_distribution = jnp.matmul(E_distribution, jnp.array([[1-P01, P01],[P10, 1-P10]]))


            '''
                401k related constants
            '''
            # 401k amount growth rate
            r_bar = 0.02
            # income fraction goes into 401k 
            yi = 0.04
            Pa = Pa[:T_max]
            Nt = [np.sum(Pa[t:]) for t in range(T_min,T_max)]
            # factor used to calculate the withdraw amount 
            Dn = [(r_bar*(1+r_bar)**N)/((1+r_bar)**N - 1) for N in Nt]
            Dn[-1] = 1
            Dn = jnp.array(Dn)
            # cash accumulated before retirement 
            nEarning = yi*E_distribution[1]*(1+jnp.dot(S_distribution,gGDP))*detEarning[:45]
            n_balance = np.zeros(T_R)
            for t in range(T_R):
                nMultiplier = jnp.array([(1+r_bar)**(t-i) for i in range(t)])
                n_balance[t] = (nEarning[:t] * nMultiplier).sum()
            # cash payouts after retirement 
            n_payout = []
            amount = n_balance[-1]
            for t in range(45, 60):
                n_payout.append(amount*Dn[t])
                amount = amount - amount*Dn[t]
                n_balance = jnp.append(n_balance,amount)
            n_payout = jnp.array(n_payout)


            '''
                housing related constants
            '''
            # variable associated with housing and mortgage 
            # age limit of buying a house
            ageLimit = 30
            mortgageLength = 30
            # mortgage rate 
            rh = 0.045
            # housing unit
            H = 1000
            # max rent unit
            Rl = 500
            # housing price constant 
            pt = 2*250/1000
            # 30k rent 1000 sf
            pr = 2*10/1000 * 2 
            # constant cost 
            c_h = 5
            c_s = H*pt*0.4
            # Dm is used to update the mortgage payment
            Dm = [(1+rh) - rh*(1+rh)**(T_max - t)/((1+rh)**(T_max-t)-1) for t in range(T_min, T_max)]
            Dm[-1] = 0
            Dm = jnp.array(Dm)
            # 30 year mortgage
            Ms = []
            M = H*pt*0.8
            m = M*(1+rh) - Dm[30]*M
            for i in range(30, T_max):
                Ms.append(M)
                M = M*(1+rh) - m
            Ms.append(0)
            Ms = jnp.array(Ms)


            '''
                Discretize the state space
                Discretize the action space 
            '''
            # actions dicretization(hp, cp, kp)
            numGrid = 20
            #As = np.array(np.meshgrid(np.linspace(0.001,0.999,numGrid), np.linspace(0,1,numGrid), [0,1])).T.reshape(-1,3)
            As = np.array(np.meshgrid([0.50], [0.25], [0,1])).T.reshape(-1,3)
            As = jnp.array(As)
            # wealth discretization
            wealthLevel = 300
            polynomialDegree = 2
            ws = jnp.linspace(0, np.power(wealthLevel,1/polynomialDegree), numGrid)**polynomialDegree
            # age of last time bought a house value only count when o = 1. 
            aBuy = np.array(range(ageLimit))
            # dimentions of the state
            dim = (ws.size, aBuy.size, nS, nE, nO, nZ)
            dimSize = len(dim)

            xgrid = np.array([[w,ab,s,e,o,z] for w in ws
                                        for ab in aBuy
                                        for s in range(nS)
                                        for e in range(nE)
                                        for o in range(nO)
                                        for z in range(nZ)]).reshape(dim + (dimSize,))

            Xs = xgrid.reshape((np.prod(dim),dimSize))
            Xs = jnp.array(Xs)
            Vgrid = np.zeros(dim + (T_max,))

            # start of function definitions
            nX = Xs.shape[0]
            nA = As.shape[0]



            '''
                Functions Definitions
            '''
            # GDP growth depending on current S state
            gGDP = jnp.array(econRate[:,0])
            #Define the earning function, which applies for both employment status and 8 econ states
            @partial(jit, static_argnums=(0,))
            def y(t, x):
                '''
                    x = [w,ab,s,e,o,z]
                    x = [0,1, 2,3,4,5]
                '''
                if t < T_R:
                    return detEarning[t] * (1+gGDP[t]) * x[3] + (1-x[3]) * welfare
                else:
                    return detEarning[-1]

            #Earning after tax and fixed by transaction in and out from 401k account 
            @partial(jit, static_argnums=(0,))
            def yAT(t,x):
                yt = y(t, x)
                if t < T_R:
                    # yi portion of the income will be put into the 401k if employed
                    return (1-tau_L)*(yt * (1-yi))*x[3] + (1-x[3])*yt
                else:
                    # t >= T_R, n/discounting amount will be withdraw from the 401k 
                    return (1-tau_R)*yt + n_payout[t-T_R]

            #Define the utility function
            @jit
            def u(c):
                return jnp.nan_to_num(x = (jnp.power(c, 1-gamma) - 1)/(1 - gamma), nan = -jnp.inf)

            #Define the bequeath function, which is a function of bequeath wealth
            @jit
            def uB(tb):
                return B*u(tb)

            #Reward function depends on the housing and non-housing consumption
            @jit
            def R(a):
                '''
                Input:
                    a = [c,b,k,h,action]
                    a = [0,1,2,3,4]
                '''
                c = a[:,0]
                b = a[:,1]
                k = a[:,2]
                h = a[:,3]
                C = jnp.power(c, alpha) * jnp.power(h, 1-alpha)
                return u(C) + (-1/((c > 0) * (b >= 0) * (k >= 0) * (h > 0)) + 1)

            # pc*qc / (ph*qh) = alpha/(1-alpha)
            @partial(jit, static_argnums=(0,))
            def feasibleActions(t, x):
                '''
                    x = [w,ab,s,e,o,z]
                    x = [0,1, 2,3,4,5]
                    a = [c,b,k,h,action]
                    a = [0,1,2,3,4]
                '''
                # owner
                sell = As[:,2]
                ab = jnp.array(x[1], dtype = jnp.int8)
                # last term is the tax deduction of the interest portion of mortgage payment
                payment = ((t-ab) > 0)*((t-ab) <= mortgageLength)*(((t<=T_R)*tau_L + (t>T_R)*tau_R)*Ms[t-ab]*rh - m)
                # this is the fire sell term, as long as we could afford the payment, do not sell
                sell = (yAT(t,x) + x[0] + payment > 0)*jnp.zeros(nA) + (yAT(t,x) + x[0] + payment <= 0)*jnp.ones(nA)
                budget1 = yAT(t,x) + x[0] + (1-sell)*payment + sell*(H*pt - Ms[t-ab] - c_s)
                h = jnp.ones(nA)*H*(1+kappa)*(1-sell) + sell*jnp.clip(budget1*As[:,0]*(1-alpha)/pr, a_max = Rl)
                c = budget1*As[:,0]*(1-sell) + sell*(budget1*As[:,0] - h*pr)
                budget2 = budget1*(1-As[:,0])
                k = budget2*As[:,1]
                k = k - (1-x[5])*(k>0)*c_k
                k = k*(1-Kc)
                b = budget2*(1-As[:,1])
                owner_action = jnp.column_stack((c,b,k,h,sell)) 


                # renter
                buy = As[:,2]*(t < ageLimit)
                budget1 = yAT(t,x) + x[0] - buy*(H*pt*0.2 + c_h)
                h = jnp.clip(budget1*As[:,0]*(1-alpha)/pr, a_max = Rl)*(1-buy) + buy*jnp.ones(nA)*H*(1+kappa)
                c = (budget1*As[:,0] - h*pr)*(1-buy) + buy*budget1*As[:,0]
                budget2 = budget1*(1-As[:,0])
                k = budget2*As[:,1]
                k = k - (1-x[5])*(k>0)*c_k
                k = k*(1-Kc)
                b = budget2*(1-As[:,1])
                renter_action = jnp.column_stack((c,b,k,h,buy))

                actions = x[4]*owner_action + (1-x[4])*renter_action
                return actions

            @partial(jit, static_argnums=(0,))
            def transition(t,a,x):
                '''
                    Input:
                        x = [w,ab,s,e,o,z]
                        x = [0,1, 2,3,4,5]
                        a = [c,b,k,h,action]
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
                nA = a.shape[0]
                s = jnp.array(x[2], dtype = jnp.int8)
                e = jnp.array(x[3], dtype = jnp.int8)
                # actions taken
                b = a[:,1]
                k = a[:,2]
                action = a[:,4]
                w_next = ((1+r_b[s])*b + jnp.outer(k,(1+r_k)).T).T.flatten().repeat(nE)
                ab_next = (1-x[4])*(t*(action == 1)).repeat(nS*nE) + x[4]*(x[1]*jnp.ones(w_next.size))
                s_next = jnp.tile(jnp.arange(nS),nA).repeat(nE)
                e_next = jnp.column_stack((e.repeat(nA*nS),(1-e).repeat(nA*nS))).flatten()
                z_next = x[5]*jnp.ones(w_next.size) + ((1-x[5]) * (k > 0)).repeat(nS*nE)
                # job status changing probability and econ state transition probability
                pe = Pe[s, e]
                ps = jnp.tile(Ps[s], nA)
                prob_next = jnp.column_stack(((1-pe)*ps,pe*ps)).flatten()
                # owner
                o_next_own = (x[4] - action).repeat(nS*nE)
                # renter
                o_next_rent = action.repeat(nS*nE)
                o_next = x[4] * o_next_own + (1-x[4]) * o_next_rent   
                return jnp.column_stack((w_next,ab_next,s_next,e_next,o_next,z_next,prob_next))

            # used to calculate dot product
            @jit
            def dotProduct(p_next, uBTB):
                return (p_next*uBTB).reshape((p_next.shape[0]//(nS*nE), (nS*nE))).sum(axis = 1)


            # define approximation of fit
            @jit
            def fit(v, xpp):
                value = vmap(partial(jnp.interp,xp = ws))(x = xpp[:,0], fp = v[:,jnp.array(xpp[:,1], dtype = int),
                                                                   jnp.array(xpp[:,2], dtype = int),
                                                                   jnp.array(xpp[:,3], dtype = int),
                                                                   jnp.array(xpp[:,4], dtype = int),
                                                                   jnp.array(xpp[:,5], dtype = int)].T)
                return jnp.nan_to_num(x = value, nan = -jnp.inf)


            @partial(jit, static_argnums=(0,))
            def V_solve(t,V_next,x):
                '''
                x = [w,ab,s,e,o,z]
                x = [0,1, 2,3,4,5]
                xp:
                    w_next    0
                    ab_next   1
                    s_next    2
                    e_next    3
                    o_next    4
                    z_next    5
                    prob_next 6
                '''
                actions = feasibleActions(t,x)
                xp = transition(t,actions,x)
                # bequeath utility, wealth level, the retirement account, heir sell the house at a cost of 25k
                TB = xp[:,0] + n_balance[t] + xp[:,4]*(H*pt-Ms[jnp.array(t-xp[:,1], dtype = jnp.int8)]*(1+rh) - 25)
                bequeathU = uB(TB)
                if t == T_max-1:
                    Q = R(actions) + beta * dotProduct(xp[:,6], bequeathU)
                else:
                    Q = R(actions) + beta * dotProduct(xp[:,6], Pa[t]*fit(V_next, xp) + (1-Pa[t])*bequeathU)
                Q = Q + (-jnp.inf)*(x[1] >= t)
                v = Q.max()
                cbkha = actions[Q.argmax()]
                return v, cbkha
            
            
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
            
            