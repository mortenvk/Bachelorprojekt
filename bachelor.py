import random
import numpy as np
import matplotlib
import numba
from numba import jit
from matplotlib import pyplot as plt



'''
class Game :  
    def __init__(self, p1, p2) -> None:
        pass

    def klein_demand(self):
        if self.p1 < self.p2:
            d = 1 - self.p1
        elif self.p1 == self.p2:
            d = 0.5 * (1 - self.p1)
        else:
            d = 0
        return d
    
    def game(prices, periods, alpha, theta):
    a = len(prices)
    Q_table = np.zeros((a, a))
    Q_table2 = np.zeros((a, a))
    profitability = 0.0
    print('CHECK', int(periods/2)-1)
    p_ipriser =np.zeros(int(periods/2)-1)
    p_jpriser =np.zeros(int(periods/2)-1)
    prev_p = np.zeros((2,2), dtype=int)
    for i in range(1):
        for j in range(1):
            prev_p[i,j] = np.random.choice(len(prices))
            #print('prev_p', prev_p)
    t = 3
    i_counter = 0
    j_counter = 0
    for t in range(t, periods+1):
        epsilon = (1-theta)**t
        
        if t % 2 != 0: 
            update(Q_table, prev_p, alpha, 0.95, prices,1)
            p_i = player3(prices, Q_table, epsilon, prev_p)
            prev_p[0,0] = prev_p[0,1]
            prev_p[0,1] = p_i
            prev_p[1,0] = prev_p[1,1]
            p_ipriser[i_counter] = (prices[p_i])
            i_counter += 1
            print('Spiller 1 tur: p:', prices[p_i],' p_j: ', prices[prev_p[1,1]],'iteration:', t,'Q_table: \n', Q_table)
            profitability += profit(prices[p_i],prices[prev_p[1,1]] ) 
            
        else: 
            update(Q_table2, prev_p, alpha, 0.95, prices, 0)
            p_j = player4(prices, Q_table2, epsilon, prev_p)
            prev_p[1,0] = prev_p[1,1]
            prev_p[1,1] = p_j
            prev_p[0,0] = prev_p[0,1]
            p_jpriser[j_counter] = (prices[p_j])
            j_counter += 1
            print('Spiller 2 tur: p:', prices[p_j], 'p_i', prices[prev_p[0,1]],' iteration: ', t,'Q_table2: \n', Q_table2)
            profitability += profit(prices[prev_p[0,1]],prices[p_j] )
    return (1/periods)*profitability, p_ipriser, p_jpriser
'''

#@jit
def demand(p1,p2):
        if p1 < p2:
            d = 1 - p1
        elif p1 == p2:
            d = 0.5 * (1 - p1)
        else:
            d = 0
        return d
    
    
x = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]

#@numba.jit(nopython=True)
def player1(prices): 
    a = np.random.choice(prices) 
    return a

#@numba.jit(nopython=True)
def player2(prices): 
    b = np.random.choice(len(prices)) 
    return b

#print(res)


#@jit
def player3(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p3 = np.random.choice(len(prices))
        #print('now its random', epsilon)
    else:
        #p3, pyt = np.unravel_index(np.argmax(Q),Q.shape)
        p3 = np.argmax(Q[:,prev[1,1]])
    return p3


#@jit
def player4(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p4 = np.random.choice(len(prices))
        #print('now its random', epsilon)
    else:
        p4 = np.argmax(Q[:,prev[0,1]])
    return p4



    
#@jit
def update(Q, prev, alpha, delta, prices, indic):
    if indic == 1: 
        p1 = prices[prev[0,0]]
        p2 = prices[prev[1,0]]
        p22 = prices[prev[1,1]]
        pe1 = Q[prev[0,0],prev[1,0]]
        ne1 = p1*demand(p1,p2) + delta* p1*demand(p1,p22) + delta**2 * Q[np.argmax(Q[:,prev[1,1]]),prev[1,1]]
        Q[prev[0,0], prev[1,0]] = (1-alpha) * pe1 + alpha * ne1
        #print('GAME player 1 ne and pe', ne1, pe1)
    else: 
        p1 = prices[prev[1,0]]
        p2 = prices[prev[0,0]]
        p22 = prices[prev[0,1]]
        pe2 = Q[prev[1,0],prev[0,0]]
        ne2 = p1*demand(p1,p2) + delta* p1*demand(p1,p22) + delta**2 * Q[np.argmax(Q[:,prev[0,1]]),prev[0,1]]
        Q[prev[1,0], prev[0,0]] = (1-alpha) * pe2 + alpha * ne2
    


#@jit    
def profit(pris1, pris2):
    return pris1*demand(pris1,pris2)



def opti(Q, lastp, prev1, prices, alpha, delta):
    tol = 1
    print('p2last price', lastp)
    print('old q_table:\n', Q)
    firstq = Q[prev1, lastp]
    while tol > 0.00001:
            print('1 iteration', tol)
            print('prev1:', prev1)
            p1 = prices[prev1]
            p2 = prices[lastp]
            pe = Q[prev1, lastp]
            oldQ = Q[prev1, lastp]
            ne = p1 * demand(p1,p2) + delta * p1 * demand(p1,p2) + delta**2 * Q[np.argmax(Q[:,lastp]),lastp]
            print('demand p1:', p1*demand(p1, p2))
            print('pe:', pe)
            print('ne:', ne)
            Q[prev1, lastp] = (1-alpha) * pe + alpha * ne
            tol = np.abs(Q[prev1, lastp] - oldQ)
            prev1 = np.argmax(Q[:,lastp])
    maxp = Q[prev1, lastp]
    opt = firstq/maxp
    print('maxp:', maxp)
    print('old:', firstq)
    print('end Q', Q)
    return opt
            


#@jit
def game(prices, periods, alpha, theta):
    a = len(prices)
    Q_table = np.zeros((a, a))
    Q_table2 = np.zeros((a, a))
    profitability = 0.0
    optimality = 0.0
    print('CHECK', int(periods/2)-1)
    p_ipriser =np.zeros(int(periods/2)-1)
    p_jpriser =np.zeros(int(periods/2)-1)
    prev_p = np.zeros((2,2), dtype=int)
    for i in range(1):
        for j in range(1):
            prev_p[i,j] = np.random.choice(len(prices))
    t = 3
    i_counter = 0
    j_counter = 0
    stepsize = periods/100
    step_counter =0
    opt_arr = np.zeros(int(periods/2/5000))
    k = 0
    for t in range(t, periods+1):
        epsilon = (1-theta)**t
        
        if t % 2 != 0: 
            update(Q_table, prev_p, alpha, 0.95, prices,1)
            p_i = player3(prices, Q_table, epsilon, prev_p)
            prev_p[0,0] = prev_p[0,1]
            prev_p[0,1] = p_i
            prev_p[1,0] = prev_p[1,1]
            p_ipriser[i_counter] = (prices[p_i])
            i_counter += 1
            #print('Spiller 1 tur: p:', prices[p_i],' p_j: ', prices[prev_p[1,1]],'iteration:', t,'Q_table: \n', Q_table)
            profitability += profit(prices[p_i],prices[prev_p[1,1]] ) 
            if step_counter == stepsize:
                print("t and stepsize", t-3, stepsize)
                opt_arr[k] = opti(Q_table, prev_p[1,1],p_i, prices,alpha, 0.95)
                step_counter = 0
                k += 1
            step_counter +=1
            
        else: 
            update(Q_table2, prev_p, alpha, 0.95, prices, 0)
            p_j = player4(prices, Q_table2, epsilon, prev_p)
            prev_p[1,0] = prev_p[1,1]
            prev_p[1,1] = p_j
            prev_p[0,0] = prev_p[0,1]
            p_jpriser[j_counter] = (prices[p_j])
            j_counter += 1
            #print('Spiller 2 tur: p:', prices[p_j], 'p_i', prices[prev_p[0,1]],' iteration: ', t,'Q_table2: \n', Q_table2)
            profitability += profit(prices[prev_p[0,1]],prices[p_j] )
    optimality = opti(Q_table, p_j, p_i, prices, alpha, 0.95)
    return (1/periods)*profitability, p_ipriser, p_jpriser, Q_table, optimality, opt_arr
            
#@numba.jit(nopython=True)     
'''def rep_games(reps): 
    i=0
    pro_arr=np.zeros(reps)   
    while i < reps:
        print('current i:', i)
        pro, p1, p2 = game( x, 1000, 0.3, 0.01372)
        u = (1/reps)*pro
        pro_arr[i] = u
        i+=1
    return pro_arr, p1, p2'''


    

#pro = game( x, 1000, 0.3, 0.01372)
#print('Profitability:', (1/1000)*pro)
#pro_arr, i, u = rep_games(1000)

pro, arr, arr1, Q_t, optimality_1, arr_opt = game( x, 500000, 0.3, 0.00002763)
t_arr = np.arange(249999)
print('profitability:', pro)
print('optimality player i:', optimality_1)
print('q_table player i:', Q_t)
print('optimality array:', arr_opt)
plt.plot(t_arr,arr,label='Player 1')
plt.plot(t_arr,arr1, label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Price")
plt.legend()
plt.show()

test = np.array([[1,2,3], [3,4,5], [4,5,6]])
''' 
t2_arr= np.arange(1000)
plt.plot(t2_arr, pro_arr, label='Avg. profitability')
plt.xlabel("Time t")
plt.ylabel("Profitability")
plt.legend()
plt.show()'''
