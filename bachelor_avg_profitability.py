import random
import numpy as np
import matplotlib
import numba
import datetime as time
from numba import jit
from matplotlib import pyplot as plt

#random.seed(123)
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
def player3(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p3 = np.random.choice(len(prices))
        #print('now its random', epsilon)
    else:
        #p3, pyt = np.unravel_index(np.argmax(Q),Q.shape)
        p3 = np.argmax(Q[:,prev[1,1]])
    return p3


#A Q-learning player 
#@jit
def player4(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p4 = np.random.choice(len(prices))
        #print('now its random', epsilon)
    else:
        p4 = np.argmax(Q[:,prev[0,1]])
    return p4



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
def opti(Q, lastp, prev1, prices, alpha, delta):
    tol = 1
    Q_here = Q
    print('p2last price', lastp)
    print('old q_table:\n', Q)
    firstq = Q_here[prev1, lastp]
    while tol > 0.00001:
            print('1 iteration', tol)
            print('prev1:', prev1)
            p1 = prices[prev1]
            p2 = prices[lastp]
            pe = Q_here[prev1, lastp]
            oldQ = Q_here[prev1, lastp]
            ne = p1 * demand(p1,p2) + delta * p1 * demand(p1,p2) + delta**2 * Q_here[np.argmax(Q_here[:,lastp]),lastp]
            '''print('demand p1:', p1*demand(p1, p2))
            print('pe:', pe)
            print('ne:', ne)'''
            Q_here[prev1, lastp] = (1-alpha) * pe + alpha * ne
            tol = oldQ - Q_here[prev1, lastp]
            prev1 = np.argmax(Q[:,lastp])
    maxp = Q_here[prev1, lastp]
    opt = firstq/maxp
    '''print('maxp:', maxp)
    print('old:', firstq)
    print('end Q', Q)'''
    return opt
            

#Running a simulation of x periods with x prices and 2 players. 
#@jit
def game(prices, periods, alpha, theta):
    a = len(prices)
    Q_table = np.zeros((a, a))
    Q_table2 = np.zeros((a, a))
    optimality = 0.0
    print('CHECK', int(periods/2)-1, 'starting a run with ', periods, ' periods')
    p_ipriser =np.zeros(int(periods/2)-1)
    p_jpriser =np.zeros(int(periods/2)-1)
    prev_p = np.zeros((2,2), dtype=int)
    prof_arr = np.zeros(int(periods-2))
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
            update(Q_table, prev_p, alpha, 0.95, prices,1)
            p_i = player3(prices, Q_table, epsilon, prev_p)
            prev_p[0,0] = prev_p[0,1]
            prev_p[0,1] = p_i
            prev_p[1,0] = prev_p[1,1]
            p_ipriser[i_counter] = (prices[p_i])
            i_counter += 1
            #print('Spiller 1 tur: p:', prices[p_i],' p_j: ', prices[prev_p[1,1]],'iteration:', t,'Q_table: \n', Q_table)
            
            prof_arr[t-3] = profit(prices[prev_p[0,1]], prices[prev_p[1,1]])
                
            '''if step_counter == stepsize:
                print('Profitability: ',profit(prices[p_i],prices[prev_p[1,1]]), ' period: ' , t, ' stepcounter, stepsize: ', step_counter, stepsize )
                prof_arr[k] = profit(prices[p_i],prices[prev_p[1,1]])
                k +=1 
                step_counter = 0'''
                
            '''if step_counter == stepsize:
                print("t and stepsize", t-3, stepsize)
                opt_arr[b] = opti(Q_table, prev_p[1,1], p_i, prices, alpha, 0.95)
                step_counter = 0
                b += 1
            step_counter +=1'''
                
                
      
        else: 
            update(Q_table2, prev_p, alpha, 0.95, prices, 0)
            p_j = player4(prices, Q_table2, epsilon, prev_p)
            #p_j = player2(prices)
            prev_p[1,0] = prev_p[1,1]
            prev_p[1,1] = p_j
            prev_p[0,0] = prev_p[0,1]
            p_jpriser[j_counter] = (prices[p_j])
            j_counter += 1
            #print('Spiller 2 tur: p:', prices[p_j], 'p_i', prices[prev_p[0,1]],' iteration: ', t,'Q_table2: \n', Q_table2)
            prof_arr[t-3] = profit(prices[prev_p[0,1]], prices[p_j])
    #optimality = opti(Q_table, p_j, p_i, prices, alpha, 0.95)
    print ('B', b)
    return prof_arr, p_ipriser, p_jpriser, Q_table, opt_arr


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
def many_games(prices, periods, alpha, theta, learners):
    total_opt_arr = np.zeros((learners,periods-2),dtype=object)
    for i in range(learners):
        print('Run #', i, ' of ', learners, ' runs.')
        proi, arri, arr1i, Q_ti, arr_opt_i = game(prices, periods, alpha, theta)
        total_opt_arr[i] = proi
    return (total_opt_arr)

'''
many_profs = many_games(x, 500000, 0.3, 0.0000276306393827805,40)
print('multi-dim prof', many_profs)

samlet_prof = many_profs.mean(0)
numpy_prof = np.mean(many_profs, axis=0)'''

#print('Average array: ', samlet_prof)
#collecting profitability from many_games. 
def prof_tests(prices, alpha, theta_list, learners):
    prof_array = np.zeros((10))
    for i in range(50000,550000,50000):
        print('i',i)
        #prof, go, go1, go2, go3 = game(prices, i, alpha, theta_list[int((i/50000)-1)])
       
        prof = many_games(prices, i, alpha, theta_list[int((i/50000)-1)],learners)

        prof_array[int((i/50000)-1)] = prof
        
    return prof_array


# List of thetas corresponding to [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]
theta_list = [(0.0002763),0.000138145562602519,0.0000920991623314899, 0.0000690751669906070, 0.0000552605153133283 ,0.0000460506414965361 , 0.0000394721082643035, 0.0000345381799382402, 0.0000307006632982448, 0.0000276306393827805]

#beta version of prof_tests
def prof_tests2(prices, periods, alpha, theta, learners):
    total_opt_arr = np.zeros((learners), dtype=object)
    
    for i in range(learners):
        proi, arri, arr1i, Q_ti, arr_opt_i = game(prices, periods, alpha, theta)
        total_opt_arr[i] = proi
        
    return total_opt_arr
    
    
###
#a random game with 500000 reps
'''
prof_arr, arr, arr1, bla, bla= game( x, 500000, 0.3, 0.0000276306393827805)
print('profitability',prof_arr)
t_arr1 = np.arange(0,499998,2)
t_arr2 = np.arange(1,449999,2)

plt.plot(t_arr1,arr,'-',label='Player 1', )
plt.plot(t_arr1,arr1,'-', label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Price")
plt.legend()
plt.show()
'''
'''
window_size = 1000
  
i = 0
# Initialize an empty list to store moving averages
moving_averages = []
# Loop through the array t o
#consider every window of size 3
while i < len(samlet_prof) - window_size + 1:
  
    # Calculate the average of current window
    window_average = np.sum(samlet_prof[i:i+window_size]) / window_size
      
    # Store the average of current
    # window in moving average list
    moving_averages.append(window_average)
      
    # Shift window to right by one position
    i += 1
'''
#print('moving averages:', moving_averages[0:100000])

###
#Plotting 2 simultaneous plots
'''fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(t_arr1,arr,'-',label='Player 1', )
ax1.plot(t_arr1,arr1,'-', label='Player 2')

ax2.plot(moving_averages,label ='profitabilitet')
ax1.set(xlabel=("Time t"), ylabel=("Price"))

ax2.set(ylim=(0.04,0.15))
plt.legend()
plt.show()'''

'''
plt.plot(moving_averages, label="Average profitability")
plt.xlabel('t')
plt.ylabel('Avg. profitability')
plt.ylim(0.00, 0.15)
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
#print('Profitability:', (1/1000)*pro)
#pro_arr, i, u = rep_games(1000)

#OPTIMALITY BEREGNING
pro, arr, arr1, Q_t, arr_opt = game( x, 500000, 0.3, 0.0000276306393827805)
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
plt.legend()
plt.show()