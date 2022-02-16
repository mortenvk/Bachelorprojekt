import random
import numpy as np

def demand(p1, p2):
    if p1 < p2:
        d = 1 - p1
    elif p1 == p2:
        d = 0.5*(1-p1)
    else:
        d = 0
    return d


x = [0.2, 0.5, 1.0]


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

def player3(prices, Q, epsilon):
    if random.uniform(0,1) < epsilon:
        p3 = np.random.choice(len(prices))
    else:
        p3, pyt = np.unravel_index(np.argmax(Q),Q_table.shape)
    return p3



#update
def update(Q, prev, alpha, delta, prices):
    p1 = prices[prev[0,0]]
    p2 = prices[prev[1,0]]
    p22 = prices[prev[1,1]]
    pe = Q_table[prev[0,0],prev[1,0]]
    ne = p1*demand(p1,p2) + delta* p1*demand(p1,p22) + delta**2 * Q_table[np.argmax(Q_table[:,prev[1,1]]),prev[1,1]]
    Q_table[prev[0,0], prev[1,0]] = (1-alpha) * pe + alpha * ne

def game(demand, prices, periods, alpha):
    prev_p = np.zeros((2,2), dtype=int)
    for i in range(1):
        for j in range(1):
            prev_p[i,j] = np.random.choice(len(prices))
            print('prev_p', prev_p)
    t = 3
    for t in range(t, periods,2):
        update(Q_table, prev_p, alpha, 0.95, prices)
        my_p = player3(prices, Q_table, 0.85)
        prev_p[1,0] = prev_p[1,1]
        prev_p[0,0] = prev_p[0,1]
        prev_p[0,1] = my_p
        prev_p[1,1] = player2(prices)
        print(prices[my_p], prices[prev_p[1,1]], t,'\n', Q_table)

game(demand, x,50, 0.3)