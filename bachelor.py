import random
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def demand(p1, p2):
    if p1 < p2:
        d = 1 - p1
    elif p1 == p2:
        d = 0.5*(1-p1)
    else:
        d = 0
    return d

x = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]


def player1(prices): 
    a = np.random.choice(prices) 
    return a

print('her', np.random.choice(len(x)))
def player2(prices): 
    b = np.random.choice(len(prices)) 
    return b

def play(demand,  prices):
    p1 = player1(prices)
    p2  = player2(prices)
    d1 = demand(p1, p2)
    return (p1, p2, d1)

#res = play(demand, x)
#print(res)
a = len(x)
Q_table = np.zeros((a, a))
Q_table2 = np.zeros((a, a))

def player3(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p3 = np.random.choice(len(prices))
        print('now its random', epsilon)
    else:
        #p3, pyt = np.unravel_index(np.argmax(Q),Q.shape)
        p3 = np.argmax(Q_table[:,prev[1,1]])
    return p3

def player4(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p4 = np.random.choice(len(prices))
        print('now its random', epsilon)
    else:
        p4 = np.argmax(Q_table2[:,prev[0,1]])
    return p4



#update
def update(Q, prev, alpha, delta, prices, indic):
    if indic == 1: 
        p1 = prices[prev[0,0]]
        p2 = prices[prev[1,0]]
        p22 = prices[prev[1,1]]
        pe = Q_table[prev[0,0],prev[1,0]]
        ne = p1*demand(p1,p2) + delta* p1*demand(p1,p22) + delta**2 * Q_table[np.argmax(Q_table[:,prev[1,1]]),prev[1,1]]
        Q_table[prev[0,0], prev[1,0]] = (1-alpha) * pe + alpha * ne
    else: 
        p1 = prices[prev[1,0]]
        p2 = prices[prev[0,0]]
        p22 = prices[prev[0,1]]
        pe = Q_table2[prev[1,0],prev[0,0]]
        ne = p1*demand(p1,p2) + delta* p1*demand(p1,p22) + delta**2 * Q_table2[np.argmax(Q_table2[:,prev[0,1]]),prev[0,1]]
        Q_table2[prev[1,0], prev[0,0]] = (1-alpha) * pe + alpha * ne
        
        
p_priser =[]
p1_priser = []
def game(demand, prices, periods, alpha, theta):
    prev_p = np.zeros((2,2), dtype=int)
    for i in range(1):
        for j in range(1):
            prev_p[i,j] = np.random.choice(len(prices))
            print('prev_p', prev_p)
    t = 3
 
    for t in range(t, periods+1):
        epsilon = (1-theta)**t
        
        if t % 2 != 0: 
            update(Q_table, prev_p, alpha, 0.95, prices,1)
            my_p = player3(prices, Q_table, epsilon, prev_p)
            prev_p[0,0] = prev_p[0,1]
            prev_p[0,1] = my_p
            prev_p[1,0] = prev_p[1,1]
            p_priser.append(prices[my_p])
            print('Spiller 1 tur: p:', prices[my_p],' p_j: ', prices[prev_p[1,1]],'iteration:', t,'Q_table: \n', Q_table)
        else: 
            update(Q_table2, prev_p, alpha, 0.95, prices, 0)
            my_p = player4(prices, Q_table2, epsilon, prev_p)
            prev_p[1,0] = prev_p[1,1]
            prev_p[1,1] = my_p
            prev_p[0,0] = prev_p[0,1]
            p1_priser.append(prices[my_p])
            print('Spiller 2 tur: p:', prices[my_p], 'p_i', prices[prev_p[0,1]],' iteration: ', t,'Q_table2: \n', Q_table2)

'''def rep_games(): 
    i=0
    while i < 500000:
        game(demand, x, 1000, 0.3, 0.01372)
    
    if 
        i+1'''
    
    
    
game(demand, x, 1000, 0.3, 0.01372)
print(len(p_priser))
print(len(p1_priser))

arr = np.array(p_priser)
arr1 = np.array(p1_priser)

t_arr = np.arange(499)

plt.plot(t_arr,arr,label='Player 1')
plt.plot(t_arr,arr1, label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Price")
plt.legend()
plt.show()

test = np.array([[1,2,3], [3,4,5], [4,5,6]])
