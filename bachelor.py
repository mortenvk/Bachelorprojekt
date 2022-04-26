import random
import numpy as np
import matplotlib
import numba
import datetime as time
from numba import jit
from matplotlib import pyplot as plt

#random.seed(1235)
#Demand function from Klein (2020)
#@jit
def demand(p1,p2):
        if p1 < p2:
            d = 1 - p1
        elif p1 == p2:
            d = 0.5 * (1 - p1)
        else:
            d = 0
        return d
    
#Price list, k= 6
x = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]

#A player picking random prices
#@numba.jit(nopython=True)
def player1(prices): 
    a = np.random.choice(prices) 
    return a

#A player picking random prices
#@numba.jit(nopython=True)
def player2(prices): 
    b = np.random.choice(len(prices)) 
    return b


#A Q-learning player 
#@jit
def player3(prices, Q, epsilon, p2):
    if random.uniform(0,1) < epsilon:
        p3 = np.random.choice(len(prices))
        #print('now its random', epsilon)
    else:
        #p3, pyt = np.unravel_index(np.argmax(Q),Q.shape)
        p3 = np.argmax(Q[:,int(p2)])
    return p3

#A Q-learning player 
def player4(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p4 = np.random.choice(len(prices))
        #print('now its random', epsilon)
    else:
        p4 = np.argmax(Q[:,prev[0,1]])
    return p4

#A restricted Q-learning player 
#@jit
def player5(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p4 = np.random.choice(len(prices))
        #print('now its random', epsilon)
    else:
        p4 = np.argmax(Q[:,prev[0,1]])
        if p4 > prev[0,1]:
            p4 = prev[0,1]
        elif p4 == 0:
            p4 = p4 + 1
    return p4

#A restricted Q-learning player 
def player6(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p4 = np.random.choice(len(prices))
        #print('now its random', epsilon)
    else:
        p4 = np.argmax(Q[:,prev[0,1]])
        if random.uniform(0,1) < 0.5:
            p4 = prev[1,1]
    return p4

#A tit for tat player
def tit4tat(prev):
    pt = prev[0,1]
    return pt







#Function updating the Q table of a player. 
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
    


#Function determining profit
#@jit    
def profit(pris1, pris2):
    return pris1*demand(pris1,pris2)



#Optimality function, keeping opponent price constant and iterates until convergence towards perfect Q-value
def opti(Q, lastp, prev1, prices, alpha, delta, theta, current_round):
    Q_here = np.copy(Q)
    firstq = Q_here[prev1, lastp]
    tol = 1
    p2 = prices[lastp]
    #print('p2:',p2)
    while tol > 0.00001:
        epsilon = (1-theta)**current_round
        p1_indic = player3(prices, Q_here, epsilon, lastp)
        p1 = prices[p1_indic]
        pe = Q_here[p1_indic, lastp]
        ne = p1 * demand(p1,p2) + delta * p1 * demand(p1,p2) + delta**2 * Q_here[np.argmax(Q_here[:,lastp]),lastp]
        Q_here[p1_indic, lastp] = (1-alpha) * pe + alpha * ne
        tol = abs(pe - Q_here[p1_indic, lastp])
        current_round+=2
    maxp = Q_here[np.argmax(Q_here[:,lastp]),lastp]
    #print('maxp:', maxp)
    #print('firstq:', firstq)
    opt = firstq/maxp
    
    if (opt > 10):
        print('maxp:', maxp, 'p2:', p2)
        print('old:', firstq)
  
    return opt
            

#Running a simulation of x periods with x prices and 2 players. 
#@jit
def game(prices, periods, alpha, theta, delta):
    a = len(prices)
    Q_table = np.zeros((a, a))
    Q_table2 = np.zeros((a, a))
    optimality = 0.0
    print('CHECK', int(periods/2)-1, 'starting a run with ', periods, ' periods')
    p_ipriser =np.zeros(int(periods/2)-1)
    p_jpriser =np.zeros(int(periods/2)-1)
    prev_p = np.zeros((2,2), dtype=int)
    prof_arr = np.zeros(int(periods-2))
    prof_arr2 = np.zeros(int(periods-2))

    for i in range(1):
        for j in range(1):
            prev_p[i,j] = np.random.choice(len(prices))
    t = 3
    i_counter = 0
    j_counter = 0
    stepsize = periods/50
    step_counter =0
    opt_arr = np.zeros(int(periods/2/5000-1))
    b = 0
    for t in range(t, periods+1):
        epsilon = (1-theta)**t
        
        if t % 2 != 0: 
            update(Q_table, prev_p, alpha, delta, prices,1)
            p_i = player3(prices, Q_table, epsilon, prev_p[1,1])
            prev_p[0,0] = prev_p[0,1]
            prev_p[0,1] = p_i
            prev_p[1,0] = prev_p[1,1]
            p_ipriser[i_counter] = (prices[p_i])
            i_counter += 1
            #print('Spiller 1 tur: p:', prices[p_i],' p_j: ', prices[prev_p[1,1]],'iteration:', t,'Q_table: \n', Q_table)
            prof_arr2[t-3] = profit(prices[prev_p[1,1]], prices[p_i])
            prof_arr[t-3] = profit(prices[prev_p[0,1]], prices[prev_p[1,1]])
        
            '''  
            if step_counter == stepsize:
                #print("t and stepsize", t-3, stepsize)
                opt_arr[b] = opti(Q_table, prev_p[1,1], p_i, prices, alpha, delta, theta, t)
                step_counter = 0
                b += 1
            step_counter +=1
            '''
                
                
      
        else: 
            update(Q_table2, prev_p, alpha, delta, prices, 0)
            #p_j = tit4tat(prev_p)
            p_j = player4(prices, Q_table2, epsilon, prev_p)
            #p_j = player2(prices)
            #p_j = player5(prices, Q_table2, epsilon, prev_p)
            #p_j = player6(prices, Q_table2, epsilon, prev_p)
            prev_p[1,0] = prev_p[1,1]
            prev_p[1,1] = p_j
            prev_p[0,0] = prev_p[0,1]
            p_jpriser[j_counter] = (prices[p_j])
            j_counter += 1
            #print('Spiller 2 tur: p:', prices[p_j], 'p_i', prices[prev_p[0,1]],' iteration: ', t,'Q_table2: \n', Q_table2)
            prof_arr[t-3] = profit(prices[prev_p[0,1]], prices[p_j])
            prof_arr2[t-3] = profit(prices[p_j], prices[prev_p[0,1]])
            step_counter +=1
    #optimality = opti(Q_table, p_j, p_i, prices, alpha, 0.95)
    print ('B', b)
    return prof_arr, p_ipriser, p_jpriser, Q_table, opt_arr, prof_arr2


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


#simulating multiple runs and averaging profit
def many_games(prices, periods, alpha, theta, learners,delta):
    total_pro_arr = np.zeros((learners,periods-2),dtype=object)
    total_pro_arr2 = np.zeros((learners,periods-2),dtype=object)
    total_opt_arr = np.zeros((learners, 49), dtype = object)
    for i in range(learners):
        print('run #',i ,'of ', learners-1 , 'runs')
        proi, arri, arr1i, Q_ti, arr_opt_i, proi2 = game(prices, periods, alpha, theta, delta)
        total_pro_arr[i] = proi
        total_pro_arr2[i] = proi2
        total_opt_arr[i] = arr_opt_i
        print('profitability1',proi[-10:])
        print('profitability1',proi2[-10:])
        print('pris1:', arri[-10:])
        print('priser2:', arr1i[-10:])
    return total_pro_arr, total_opt_arr, total_pro_arr2



many_profs, many_opt, many_profs2 = many_games(x, 500000, 0.3, 0.0000276306393827805,40, 0.95)
#print('multi-dim prof', many_profs)
print('many_opt:',many_opt)

samlet_prof = many_profs.mean(0)
samlet_prof2 = many_profs2.mean(0)
samlet_opt = np.mean(many_opt,axis=0)


'''
# List of thetas corresponding to [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
theta_list = [(0.0002763),0.000138145562602519,0.0000920991623314899, 0.0000690751669906070, 0.0000552605153133283 ,0.0000460506414965361 , 0.0000394721082643035, 0.0000345381799382402, 0.0000307006632982448, 0.0000276306393827805]
    
###
#a random game with 500000 reps

''' 
'''
prof_arr, arr, arr1, q_table, bla= game( x, 500000, 0.3, 0.0000276306393827805, 0.95)
print('profitability',prof_arr[-10:])
t_arr1 = np.arange(0,499998,2)
t_arr2 = np.arange(1,499999,2)
print('pris1:', arr[-10:])
print('priser2:', arr1[-10:])

plt.plot(t_arr1,arr,'--o',label='Player 1', )
plt.plot(t_arr2,arr1,'s--', label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Price")
plt.legend()
plt.show()
'''

window_size = 1000
  
i = 0
# Initialize an empty list to store moving averages
moving_averages = []
moving_averages2 = []
# Loop through the array t o
#consider every window of size 3
while i < len(samlet_prof) - window_size + 1:
  
    # Calculate the average of current window
    window_average = np.sum(samlet_prof[
      i:i+window_size]) / window_size
    window_average2 = np.sum(samlet_prof2[
      i:i+window_size]) / window_size
      
    # Store the average of current
    # window in moving average list
    moving_averages.append(window_average)
    moving_averages2.append(window_average2)

      
    # Shift window to right by one position
    i += 1

#print(moving_averages)

t_arr1 = np.arange(0,498999)
t_arr2 = np.arange(0,498999)
plt.plot(t_arr1,moving_averages,'-',label='Player 1', )
#plt.plot(t_arr2,moving_averages2,'-', label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Profitability")
plt.ylim(0.00,0.15)
plt.legend()
plt.show()

###
#Plotting 2 simultaneous plots
'''
fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(t_arr1,arr,'-',label='Player 1', )
ax1.plot(t_arr1,arr1,'-', label='Player 2')

ax2.plot(moving_averages,label ='profitabilitet')
ax1.set(xlabel=("Time t"), ylabel=("Price"))

ax2.set(ylim=(0.04,0.15))
plt.legend()
plt.show()
'''
'''
fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(t_arr1,arr,'-',label='Player 1', )
ax1.plot(t_arr1,arr1,'-', label='Player 2')

ax2.plot(moving_averages,label ='profitabilitet')
ax1.set(xlabel=("Time t"), ylabel=("Price"))

ax2.set(ylim=(0.04,0.15))
plt.legend()
plt.show()
'''
'''
plt.plot(samlet_opt, label="Average optimality")
plt.xlabel('t')
plt.ylabel('Avg. profitability')
plt.ylim(0, 1)
plt.show()
#Printing average profitability across 10 learners and 10 different T
'''

'''
# PRINTING for EACH period switching between players
pro, arr, arr1, Q_t, optimality_1 = game( x, 500000, 0.3, 0.0000276306393827805)
t_arr1 = np.arange(249999)
print('profitability:', pro)
print('optimality player i:', optimality_1)
print('q_table player i:', Q_t)
plt.plot(t_arr1,arr,'-',label='Player 1', )
plt.plot(t_arr1,arr1,'-', label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Price")
plt.legend()
plt.show()

#print('HEY! final profitablity arr: ', pro_print)'''
'''bla, p1_prices, p2_prices, basd, jsdhf, prof_arr1 = game( x, 100000, 0.3, 0.0001381)
t_arr = np.arange(0, )

plt.plot(t_arr,prof_arr1,label='Player 1')
#plt.plot(t_arr,p2_prices, label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Price")
plt.legend()
plt.show()'''
'''
pro, arr, arr1, Q_t, arr_opt = game( x, 500000, 0.3, 0.0000276306393827805, 0.95)
print('optimality array single:', arr_opt)
'''


'''
#OPTIMALITY BEREGNING
pro, arr, arr1, Q_t, arr_opt = game( x, 500000, 0.3, 0.0000276306393827805, 0.95)
#hej = many_games(x, 500000, 0.3, 0.00002763, 3)
#print('array with 1000 learners for 500000 periods:', hej)

print('profitability:', pro)
#print('optimality player i:', optimality_1)
print('q_table player i:', Q_t)
print('optimality array:', arr_opt)

t_arr = np.arange(249999)

plt.plot(t_arr,arr,label='Player 1')
plt.plot(t_arr,arr1, label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Price")
plt.show()
'''