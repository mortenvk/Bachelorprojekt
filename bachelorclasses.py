import random
from sys import exec_prefix
import numpy as np
import matplotlib
import numba
from numba import jit
from matplotlib import pyplot as plt

class Game:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        
'''
Player class, the player can be either completely random or somewhat random (choosing between exploration and exploitation.

The player has the following attributes.
player_type: determining the type of player (either ran, ee_i or ee_j)

'''
class Player(Game):     
    def __init__(self, player_type, Q_table, prices):
        self.player_type = player_type
        self.Q_table = Q_table
        

    def player_choice(self, epsilon, prev):
        if self.player_type == 'ran':
            p = np.random.choice(len(self.prices))  
        elif self.player_type == 'ee_i': 
            if random.uniform(0,1) < epsilon:
                p = np.random.choice(len(self.prices))
        
            else:
                p = np.argmax(self.Q_table[:,prev[1,1]])
        else: 
            if random.uniform(0,1) < self.epsilon:
                p = np.random.choice(len(self.prices))
            else:
                p = np.argmax(self.Q_table[:,prev[0,1]])
        return p

    def update(self, prev, alpha, delta, prices, indic):   
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
            tol = oldQ - Q[prev1, lastp]
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
    final_profitability = 0.0
    optimality = 0.0
    print('CHECK', int(periods/2)-1, 'starting a run with ', periods, ' periods')
    p_ipriser =np.zeros(int(periods/2)-1)
    p_jpriser =np.zeros(int(periods/2)-1)
    prev_p = np.zeros((2,2), dtype=int)
    for i in range(1):
        for j in range(1):
            prev_p[i,j] = np.random.choice(len(prices))
    t = 3
    i_counter = 0
    j_counter = 0
    
    stepsize = periods/40
    step_counter =0
    opt_arr = np.zeros(int(periods/2/5000-1))
    prof_arr = np.zeros(int(periods/2/2500-1))
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
            
            
            
            if t > (periods+1)-1000:
                final_profitability += profit(prices[p_i],prices[prev_p[1,1]] )
                
      
        else: 
            update(Q_table2, prev_p, alpha, 0.95, prices, 0)
            p_j = player4(prices, Q_table2, epsilon, prev_p)
            prev_p[1,0] = prev_p[1,1]
            prev_p[1,1] = p_j
            prev_p[0,0] = prev_p[0,1]
            p_jpriser[j_counter] = (prices[p_j])
            j_counter += 1
            #print('Spiller 2 tur: p:', prices[p_j], 'p_i', prices[prev_p[0,1]],' iteration: ', t,'Q_table2: \n', Q_table2)
            if t > (periods+1)-1000:
                final_profitability += profit(prices[prev_p[0,1]],prices[p_j])
    #optimality = opti(Q_table, p_j, p_i, prices, alpha, 0.95)
    return (1/1000)*final_profitability, p_ipriser, p_jpriser, Q_table, opt_arr
            


def many_games(prices, periods, alpha, theta, learners):
    total_opt_arr = np.zeros((learners), dtype=object)
    for i in range(learners):
        proi, arri, arr1i, Q_ti, arr_opt_i = game(prices, periods, alpha, theta)
        total_opt_arr[i] = arr_opt_i
    return total_opt_arr

#tanken var her at forsøge at lave average over forskellige antal perioder
#det var dog at regne 10 forskellige thetaer som gjorde at det ikke var muligt...
'''def prof_tests(prices, alpha, theta):
    prof_array = np.zeroes((10))
    for i in range(500000,500000,50000):
        prof, go, go1, go2, go3, go4 = game(prices, i, alpha, theta)
        prof_array[(i/50000)] = prof
    return prof_array'''
# List of thetas corresponding to [50000, 100000, 150000,]
theta_list = [(0.0002763),0.000138145562602519,0.0000690751669906070, 0]

def prof_tests2(prices, periods, alpha, theta, learners):
    total_opt_arr = np.zeros((learners), dtype=object)
    for i in range(learners):
        proi, arri, arr1i, Q_ti, arr_opt_i = game(prices, periods, alpha, theta)
        total_opt_arr[i] = proi
    return total_opt_arr
    

pro_print = prof_tests2(x, 100000, 0.3, 0.0001381, 100)
t_arr = np.arange(100)

plt.plot(t_arr,pro_print,label='Profitability')

plt.xlabel("Runs")
plt.ylabel("Profitability")
plt.legend()
plt.show()

print('HEY! final profitablity arr: ', pro_print)
