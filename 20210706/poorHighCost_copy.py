import numpy as np
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import warnings
from jax import jit, random, vmap
from tqdm import tqdm
warnings.filterwarnings("ignore")
np.printoptions(precision=2)

# time line
T_min = 0
T_max = 60
T_R = 45
# discounting factor
beta = 1/(1+0.02)
# utility function parameter 
gamma = 4
# relative importance of housing consumption and non durable consumption 
alpha = 0.7
# parameter used to calculate the housing consumption 
kappa = 0.3
# uB associated parameter
B = 2
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
# deterministic income
detEarning = jnp.array(np.load("constant/detEarningHigh.npy"))
# rescale the deterministic income
detEarning = detEarning 
####################################################################################### low skill feature
detEarning = jnp.concatenate([detEarning[:46]*0.5, detEarning[46:]-45])
# Define transition matrix of economical states S
Ps = np.genfromtxt('constant/Ps.csv',delimiter=',')
fix = (np.sum(Ps, axis = 1) - 1)
for i in range(nS):
    for j in range(nS):
        if Ps[i,j] - fix[i] > 0:
            Ps[i,j] = Ps[i,j] - fix[i]
            break
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
    401k related constants
'''
# some variables associated with 401k amount
r_bar = 0.02
Pa = Pa[:T_max]
Nt = [np.sum(Pa[t:]) for t in range(T_min,T_max)]
#Factor used to calculate the withdraw amount 
Dn = [(r_bar*(1+r_bar)**N)/((1+r_bar)**N - 1) for N in Nt]
Dn[-1] = 1
Dn = jnp.array(Dn)
# income fraction goes into 401k 
yi = 0.04



'''
    housing related constants
'''
# variable associated with housing and mortgage 
# mortgage rate 
rh = 0.045
# housing unit
H = 600
# rent unit
Rl = 300
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
    M = M*(1+rh) - m
    Ms.append(M)
Ms[-1] = 0
Ms = jnp.array(Ms)

############################################################################################################ low skill feature 
# stock transaction fee
Kc = 0.02

# stock participation cost
c_k = 20


'''
    Discretize the state space
    Discretize the action space 
'''
# actions dicretization(hp, cp, kp)
numGrid = 20
As = np.array(np.meshgrid(np.linspace(0.001,0.999,numGrid), np.linspace(0,1,numGrid), [0,1])).T.reshape(-1,3)
As = jnp.array(As)
# wealth discretization 
ws = np.linspace(0, 400, 20)
ns = np.linspace(0, 300, 10)
ms = np.linspace(0, 0.8*H*pt, 10)
# scales associated with discretization
scaleW = ws.max()/ws.size
scaleN = ns.max()/ns.size
scaleM = ms.max()/ms.size

# dimentions of the state
dim = (ws.size, ns.size, ms.size, nS, nE, nO, nZ)
dimSize = len(dim)

xgrid = np.array([[w,n,m,s,e,o,k] for w in ws
                            for n in ns
                            for m in ms
                            for s in range(nS)
                            for e in range(nE)
                            for o in range(nO)
                            for k in range(nZ)]).reshape(dim + (dimSize,))

Xs = xgrid.reshape((np.prod(dim),dimSize))
Xs = jnp.array(Xs)

Vgrid = np.zeros(dim + (T_max,))
cgrid = np.zeros(dim + (T_max,))
bgrid = np.zeros(dim + (T_max,))
kgrid = np.zeros(dim + (T_max,))
hgrid = np.zeros(dim + (T_max,))
agrid = np.zeros(dim + (T_max,))

# start of function definitions
nX = Xs.shape[0]
nA = As.shape[0]

#Define the earning function, which applies for both employment status and 8 econ states
@partial(jit, static_argnums=(0,))
def y(t, x):
    '''
        x = [w,n,m,s,e,o,z]
        x = [0,1,2,3,4,5,6]
    '''
    if t <= T_R:
        return detEarning[t] * (1+gGDP[jnp.array(x[3], dtype = jnp.int8)]) * x[4] + (1-x[4]) * welfare
    else:
        return detEarning[-1]
    
#Earning after tax and fixed by transaction in and out from 401k account 
@partial(jit, static_argnums=(0,))
def yAT(t,x):
    yt = y(t, x)
    if t <= T_R:
        # yi portion of the income will be put into the 401k if employed
        return (1-tau_L)*(yt * (1-yi))*x[4] + (1-x[4])*yt
    else:
        # t > T_R, n/discounting amount will be withdraw from the 401k 
        return (1-tau_R)*yt + x[1]*Dn[t]
    
#Define the evolution of the amount in 401k account 
@partial(jit, static_argnums=(0,))
def gn(t, x, r = r_bar):
    if t <= T_R:
        # if the person is employed, then yi portion of his income goes into 401k 
        n_cur = x[1] + y(t, x) * yi * x[4]
    else:
        # t > T_R, n*Dn amount will be withdraw from the 401k 
        n_cur = x[1] - x[1]*Dn[t]
        # the 401 grow with the rate r 
    return (1+r)*n_cur

#Define the utility function
@jit
def u(c):
    return (jnp.power(c, 1-gamma) - 1)/(1 - gamma)

#Define the bequeath function, which is a function of bequeath wealth
@jit
def uB(tb):
    return B*u(tb)

#Reward function depends on the housing and non-housing consumption
@jit
def R(x,a):
    '''
    Input:
        x = [w,n,m,s,e,o,z]
        x = [0,1,2,3,4,5,6]
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
    # owner
    sell = As[:,2]
    
# last term is the tax deduction of the interest portion of mortgage payment
    payment = (x[2] > 0)*(((t<=T_R)*tau_L + (t>T_R)*tau_R)*x[2]*rh - m)
    
#     # if the agent is able to pay
#     if yAT(t,x) + x[0] + payment > 0:
#         sell = jnp.zeros(nA)
#         budget1 = yAT(t,x) + x[0] + (1-sell)*payment
#     # if the agent is not able to pay (force sell)
#     else:
#         sell = jnp.ones(nA)
#         budget1 = yAT(t,x) + x[0] + sell*(H*pt - x[2] - c_s)
        
    sell = (yAT(t,x) + x[0] + payment > 0)*jnp.zeros(nA) + (yAT(t,x) + x[0] + payment <= 0)*jnp.ones(nA)
    budget1 = yAT(t,x) + x[0] + (1-sell)*payment + sell*(H*pt - x[2] - c_s)
    
    # last term is the tax deduction of the interest portion of mortgage payment    
    h = jnp.ones(nA)*H*(1+kappa)*(1-sell) + sell*jnp.clip(budget1*As[:,0]*(1-alpha)/pr, a_max = Rl)
    c = budget1*As[:,0]*(1-sell) + sell*(budget1*As[:,0] - h*pr)
    budget2 = budget1*(1-As[:,0])
    k = budget2*As[:,1]*(1-Kc)
    k = k - (1-x[6])*(k>0)*c_k
    b = budget2*(1-As[:,1])
    owner_action = jnp.column_stack((c,b,k,h,sell)) 
    
    
    # renter
    buy = As[:,2]*(t < 30)
    budget1 = yAT(t,x) + x[0] - buy*(H*pt*0.2 + c_h)
    h = jnp.clip(budget1*As[:,0]*(1-alpha)/pr, a_max = Rl)*(1-buy) + buy*jnp.ones(nA)*H*(1+kappa)
    c = (budget1*As[:,0] - h*pr)*(1-buy) + buy*budget1*As[:,0]
    budget2 = budget1*(1-As[:,0])
    k = budget2*As[:,1]*(1-Kc)
    k = k - (1-x[6])*(k>0)*c_k
    b = budget2*(1-As[:,1])
    renter_action = jnp.column_stack((c,b,k,h,buy))
    
    actions = x[5]*owner_action + (1-x[5])*renter_action
    return actions

@partial(jit, static_argnums=(0,))
def transition(t,a,x):
    '''
        Input:
            x = [w,n,m,s,e,o,z]
            x = [0,1,2,3,4,5]
            a = [c,b,k,h,action]
            a = [0,1,2,3,4]
        Output:
            w_next
            n_next
            m_next
            s_next
            e_next
            o_next
            z_next
            
            prob_next
    '''
    nA = a.shape[0]
    s = jnp.array(x[3], dtype = jnp.int8)
    e = jnp.array(x[4], dtype = jnp.int8)
    # actions taken
    b = a[:,1]
    k = a[:,2]
    action = a[:,4]
    w_next = ((1+r_b[s])*b + jnp.outer(k,(1+r_k)).T).T.flatten().repeat(nE)
    n_next = gn(t, x)*jnp.ones(w_next.size)
    s_next = jnp.tile(jnp.arange(nS),nA).repeat(nE)
    e_next = jnp.column_stack((e.repeat(nA*nS),(1-e).repeat(nA*nS))).flatten()
    z_next = x[6]*jnp.ones(w_next.size) + ((1-x[6]) * (k > 0)).repeat(nS*nE)
    # job status changing probability and econ state transition probability
    pe = Pe[s, e]
    ps = jnp.tile(Ps[s], nA)
    prob_next = jnp.column_stack(((1-pe)*ps,pe*ps)).flatten()
    
    # owner
    m_next_own = ((1-action)*jnp.clip(x[2]*(1+rh) - m, a_min = 0)).repeat(nS*nE)
    o_next_own = (x[5] - action).repeat(nS*nE)
    # renter
    m_next_rent = (action*H*pt*0.8).repeat(nS*nE)
    o_next_rent = action.repeat(nS*nE)
    
    m_next = x[5] * m_next_own + (1-x[5]) * m_next_rent
    o_next = x[5] * o_next_own + (1-x[5]) * o_next_rent   
    return jnp.column_stack((w_next,n_next,m_next,s_next,e_next,o_next,z_next,prob_next))

# used to calculate dot product
@jit
def dotProduct(p_next, uBTB):
    return (p_next*uBTB).reshape((p_next.shape[0]//(nS*nE), (nS*nE))).sum(axis = 1)

# define approximation of fit
@jit
def fit(v, xp):
    return map_coordinates(v,jnp.vstack((xp[:,0]/scaleW,
                                                      xp[:,1]/scaleN,
                                                      xp[:,2]/scaleM,
                                                      xp[:,3],
                                                      xp[:,4],
                                                      xp[:,5],
                                                      xp[:,6])),
                                                     order = 1, mode = 'nearest')

@partial(jit, static_argnums=(0,))
def V(t,V_next,x):
    '''
    x = [w,n,m,s,e,o,z]
    x = [0,1,2,3,4,5]
    xp:
        w_next    0
        n_next    1
        m_next    2
        s_next    3
        e_next    4
        o_next    5
        z_next    6
        prob_next 7
    '''
    actions = feasibleActions(t,x)
    xp = transition(t,actions,x)
    # bequeath utility
    TB = xp[:,0]+x[1]*(1+r_bar)+xp[:,5]*(H*pt-x[2]*(1+rh)-25)
    bequeathU = uB(TB)
    if t == T_max-1:
        Q = R(x,actions) + beta * dotProduct(xp[:,7], bequeathU)
    else:
        Q = R(x,actions) + beta * dotProduct(xp[:,7], Pa[t]*fit(V_next, xp) + (1-Pa[t])*bequeathU)
    Q = jnp.nan_to_num(Q, nan = -jnp.inf)
    v = Q.max()
    cbkha = actions[Q.argmax()]
    return v, cbkha


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
    
    
####################################################################################### solving the model
# for t in tqdm(range(T_max-1,T_min-1, -1)):
#     if t == T_max-1:
#         v,cbkha = vmap(partial(V,t,Vgrid[:,:,:,:,:,:,:,t]))(Xs)
#     else:
#         v,cbkha = vmap(partial(V,t,Vgrid[:,:,:,:,:,:,:,t+1]))(Xs)
#     Vgrid[:,:,:,:,:,:,:,t] = v.reshape(dim)
#     cgrid[:,:,:,:,:,:,:,t] = cbkha[:,0].reshape(dim)
#     bgrid[:,:,:,:,:,:,:,t] = cbkha[:,1].reshape(dim)
#     kgrid[:,:,:,:,:,:,:,t] = cbkha[:,2].reshape(dim)
#     hgrid[:,:,:,:,:,:,:,t] = cbkha[:,3].reshape(dim)
#     agrid[:,:,:,:,:,:,:,t] = cbkha[:,4].reshape(dim)
    
# np.save("poorHigh",Vgrid)
