from audioop import avg
import random
import numpy as np
import matplotlib
import numba
import time

from numba import jit, prange
from numba import config, njit, threading_layer, set_num_threads
from matplotlib import pyplot as plt

start_time = time.time()
#random.seed(1235)
#Demand function from Klein (2020)
@njit
def demand(p1,p2):
        if p1 < p2:
            d = 1 - p1
        elif p1 == p2:
            d = 0.5 * (1 - p1)
        else:
            d = 0
        return d
    
#Price list, k= 6
x = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6, 1])

#A player picking random prices
#@numba.jit(nopython=True)
@njit
def player1(prices): 
    a = np.random.choice(prices) 
    return a

#A player picking random prices
@njit
def player2(prices): 
    b = np.random.choice(len(prices)) 
    return b


#A Q-learning player 
@jit
def player3(prices, Q, epsilon, p2):
    if random.uniform(0,1) < epsilon:
        p3 = int(np.random.choice(len(prices)))
        #print('now its random', epsilon)
    else:
        #p3, pyt = np.unravel_index(np.argmax(Q),Q.shape)
        p3 = int(np.argmax(Q[:,p2]))
    return p3

#A Q-learning player 
@jit
def player4(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p4 = int(np.random.choice(len(prices)))
        #print('now its random', epsilon)
    else:
        p4 = int(np.argmax(Q[:,prev[0,1]]))
    #print('type:', type(p4))
    return p4

#A restricted Q-learning player 
@njit
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
@njit
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
@njit
def tit4tat(prev):
    pt = prev[0,1]
    return pt

@njit
def gamma_player(prices, Q, epsilon, prev):
    if random.uniform(0,1) < epsilon:
        p4 = np.random.choice(len(prices))
        #print('now its random', epsilon)
        if p4 == len(prices)-1:
            p4 -= 1
        elif p4 == 0:
            p4 = p4 + 1
    else:
        p4 = np.argmax(Q[:,prev[0,1]])
        if p4 == len(prices)-1:
            p4 -= 1
        elif p4 == 0:
            p4 = p4 + 1  
    return p4
        
        
            





#Function updating the Q table of a player. 
@njit
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
        
@njit
def present_update(Q, prev, alpha, prices):
    p1 = prices[prev[1,0]]
    p2 = prices[prev[0,0]]
    p22 = prices[prev[0,1]]
    pe2 = Q[prev[1,0],prev[0,0]]
    ne2 = p1*demand(p1,p2)  
    Q[prev[1,0], prev[0,0]] = (1-alpha) * pe2 + alpha * ne2

#Function determining profit
@njit    
def profit(pris1, pris2):
    return pris1*demand(pris1,pris2)



#Optimality function, keeping opponent price constant and iterates until convergence towards perfect Q-value
def opti(Q, Q2, lastp, prev1, prices, alpha, delta, theta, current_round):
    Q_here = np.copy(Q)
    init_q = Q
    firstq = Q_here[prev1, lastp]
    tol = 1
    p1_indic = prev1
    p2_indic = lastp
    p2 = prices[lastp]
    #print('p2:',p2)
    while tol > 0.0001:
        epsilon = (1-theta)**current_round
        p1 = prices[p1_indic]
        p2_1_indic = np.argmax(Q2[:,p1_indic])
        #p2_1_indic = player3(prices, Q2, epsilon, p1_indic)
        p2_1 = prices[p2_1_indic]
        pe = Q_here[p1_indic, p2_indic]
        ne = p1 * demand(p1,p2) + delta * p1 * demand(p1,p2_1) + delta**2 * Q_here[np.argmax(Q_here[:,p2_1_indic]),p2_1_indic]
        Q_here[p1_indic, p2_indic] = (1-alpha) * pe + alpha * ne
        p2 = p2_1
        p2_indic = p2_1_indic
        current_round+=2
        p1_indic = player3(prices, Q_here, epsilon, int(p2_indic))
     
        tol = abs(pe - Q_here[p1_indic, p2_indic])
        
    finalp1 = np.argmax(Q_here[:,lastp])  
    maxp = Q_here[finalp1, lastp]
    #print('maxp:', maxp)
    #print('firstq:', firstq)
    opt = firstq/maxp
    print('convergence diff:', init_q-Q_here)
    
    if (opt > 1):
        print("PERIOD", current_round)
        print('maxp:', maxp, 'p2:', p2)
        print('old:', firstq, 'p1start: ', prices[prev1], 'p1 slut:', prices[finalp1])
  
    return opt
'''            

#Optimality function, keeping opponent price constant and iterates until convergence towards perfect Q-value
def t_opti(Q, Q2, lastp, prev1, prices, alpha, delta, theta, current_round):
    Q_here = np.copy(Q)
    firstq = Q_here[prev1, lastp]
    p2_indic = lastp
    tol = 1 
    print('Q values player 1 given p2 before convergence:', Q_here[:, lastp])
    for i in range(len(prices)):
        while tol > 0.0001:
            p1 = prices[i]
            p2 = prices[p2_indic]
            p2_1_indic = np.argmax(Q2[:,i])
            p2_1 = prices[p2_1_indic]
            pe = Q_here[i, p2_indic]
            ne = p1 * demand(p1,p2) + delta * p1 * demand(p1,p2_1) + delta**2 * Q_here[np.argmax(Q_here[:,p2_1_indic]),p2_1_indic]
            Q_here[i, p2_indic] = (1-alpha) * pe + alpha * ne
            p2_indic = p2_1_indic
            current_round+=2
            tol = abs(pe - Q_here[i, p2_indic])
            
    finalp1 = np.argmax(Q_here[:,lastp])  
    print('Q values player 1 given p2:', Q_here[:, lastp])
    maxp = Q_here[finalp1, lastp]
    opt = firstq/maxp
    
    if (opt > 1):
        print("PERIOD", current_round)
        print('maxp:', maxp, 'p2:', p2)
        print('old:', firstq, 'p1start: ', prices[prev1], 'p1 slut:', prices[finalp1])
  
    return opt
            
'''



#Running a simulation of x periods with x prices and 2 players. 
@jit(nopython=True)
def game(prices, periods, alpha, theta, delta):
    a = len(prices)
    Q_table = np.zeros((a, a))
    Q_table2 = np.zeros((a, a))
    optimality = 0.0
    #print('CHECK', int(periods/2)-1, 'starting a run with ', periods, ' periods')
    p_ipriser =np.zeros(int(periods/2)-1)
    p_jpriser =np.zeros(int(periods/2)-1)
    prev_p = np.zeros((2,2), dtype=numba.int64)
    prof_arr = np.zeros(int(periods-2))
    prof_arr2 = np.zeros(int(periods-2))

    for i in range(1):
        for j in range(1):
            prev_p[i,j] = np.random.choice(len(prices))
    t = 3
    i_counter = 0
    j_counter = 0
    stepsize = periods/50
    step_counter = 0
    opt_arr = np.zeros(int(periods/2/5000-1))
    b = 0
    unchanged = 0
    change1 = np.zeros(len(prices))
    change2 = np.zeros(len(prices))
    temp_br1 = np.zeros(len(prices))
    temp_br2 = np.zeros(len(prices))
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
                    opt_arr[b] = opti(Q_table, Q_table2, prev_p[1,1], prev_p[0,1], prices, alpha, delta, theta, t)
                    step_counter = 0
                    b += 1
            step_counter +=1
            '''
     
                
                
      
        else: 
            update(Q_table2, prev_p, alpha, delta, prices, 0)
            #present_update(Q_table2, prev_p, alpha, prices)
            #p_j = tit4tat(prev_p)
            p_j = player4(prices, Q_table2, epsilon, prev_p)
            #p_j = player2(prices)
            #p_j = player6(prices, Q_table2, epsilon, prev_p)
            #p_j = gamma_player(prices, Q_table2, epsilon, prev_p)
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
        '''     
        if t == 400000:
            for i in range(len(prices)):
                change1[i] = int(np.argmax(Q_table[:,i]))
                change2[i] = int(np.argmax(Q_table2[:,i]))
        elif t > 400000:
            for i in range(len(prices)):
                temp_br1[i] = int(np.argmax(Q_table[:,i]))
                temp_br2[i] = int(np.argmax(Q_table2[:,i]))
            if (np.array_equal(temp_br1,change1) == False) or (np.array_equal(temp_br2,change2) == False):
                unchanged = 1
                #print('first check', (np.array_equal(temp_br1,change1) == False), temp_br1, change1)
                #print('second check',(np.array_equal(temp_br2,change2) == False), temp_br2, change2)
        '''        
    print('argmax arrays', temp_br1, change1,(np.array_equal(temp_br1,change1) == False) )
    print('argmax arrays', temp_br2, change2, (np.array_equal(temp_br2,change2) == False))
                
    #optimality = opti(Q_table, p_j, p_i, prices, alpha, 0.95)
    #print ('B', b)
    
    #print('q_table2', Q_table2)
    return prof_arr, p_ipriser, p_jpriser, Q_table2, opt_arr, prof_arr2, unchanged



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
#@njit
def many_games(prices, periods, alpha, theta, learners,delta):
    total_pro_arr = np.zeros((learners,periods-2),dtype=np.ndarray)
    total_pro_arr2 = np.zeros((learners,periods-2),dtype=np.ndarray)
    total_opt_arr = np.zeros((learners, 49), dtype = np.ndarray)
    avg_profit = np.zeros(learners)
    avg_profit2 = np.zeros(learners)
    change_arr = np.zeros(learners)
    for i in range(learners):
        print('run #',i+1 ,'of ', learners , 'runs')
        proi, arri, arr1i, Q_ti, arr_opt_i, proi2, changes = game(prices, periods, alpha, theta, delta)
        total_pro_arr[i] = proi
        total_pro_arr2[i] = proi2
        total_opt_arr[i] = arr_opt_i
        avg_profit[i] = np.mean(proi[-10000:])
        avg_profit2[i] = np.mean(proi2[-10000:])
        change_arr[i] = changes
        #print('avg profit firm 1 & 2', avg_profit, avg_profit2, 'længde', len(avg_profit2))
        #print('profitability1',proi[-10:])
        #print('profitability1',proi2[-10:])
        #print('pris1:', arri[-10:])
        #print('priser2:', arr1i[-10:])
    return total_pro_arr, total_opt_arr, total_pro_arr2, avg_profit, avg_profit2, change_arr


def delta_prof(avg_array1, avg_array2):
    together_array = np.vstack((avg_array1, avg_array2))
    together_array = np.mean(together_array, axis=0)
    delta_1 = np.zeros(len(together_array))
    for i in range(len(together_array)):
        delta_1[i] = ((together_array[i]) / (0.125))
    return delta_1
    



#Function needed to determine the profit of the last 1000 runs - heatmap
def end_prof(p1_prof, p2_prof, avg_array1, avg_array2):
    end_prof1 = np.mean(np.array(([i[-1000:] for i in p1_prof])), axis=1)
    end_prof2 = np.mean(np.array(([i[-1000:] for i in p2_prof])), axis=1)
    together_array = np.vstack((avg_array1, avg_array2))
    together_array = np.mean(together_array, axis=0)
    
    return end_prof1, end_prof2, together_array


many_profs, many_opt, many_profs2, delta_arr, delta_arr2, change_yes = many_games(x, 500000, 0.3, 0.0000276306393827805, 1000, 0.95)
#print('multi-dim prof', many_profs)
#print('many_opt:',many_opt)
#firm1, firm2 = end_prof(many_profs, many_profs2)
delta_done1= delta_prof(delta_arr, delta_arr2)
print(delta_done1[-10:])
unique, counts = np.unique(change_yes, return_counts=True)
print(np.asarray((unique, counts)).T)


#code to plot convergence of the runs 
'''
plt.plot(change_yes, '.', label = 'Convergence')

print()
#plt.plot(delta_done1, '.', label = 'collective delta')
plt.show()
'''

#Heatmap very similar to Klein heatmap
'''
heatmap, xedges, yedges = np.histogram2d(firm1, firm2, bins=12)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
cb = plt.colorbar()
cb.set_label('profit')
plt.show()
##ax = sns.heatmap((firm1, firm2), linewidth=0.5)
#plt.show()
'''

print('starting mean calculation')
meantime = time.time()


#Calculating the mean of profitability arrays and optimality
#Can unfortunately not be optimized, as the axis argument is not supported for numba
def prof_means(prof_arr1, prof_arr2, opt_arr, ):
    return np.mean(prof_arr1, axis=0), np.mean(prof_arr2, axis=0), np.mean(opt_arr, axis=0)

samlet_prof, samlet_prof2, samlet_opt_arr = prof_means(many_profs, many_profs2, many_opt)

meanendtime = time.time()
print('ending mean. time: ', (meanendtime - meantime))

#samlet_prof2 = np.mean(many_profs2, axis=0)
#samlet_opt = np.mean(many_opt,axis=0)

window_size = 1000
  

# Initialize an empty list to store moving averages

# Loop through the array t o
#consider every window of size 3
print('starting moving avg')
avgtime = time.time()


#Function to calculate the moving average of profitability: 
#@njit
def moving_avg(fst_arr, snd_arr, window_size):

    moving_averages = []
    moving_averages2 = []
    i = 0
    while i < len(fst_arr) - window_size + 1:
    
        # Calculate the average of current window
        window_average = np.sum(fst_arr[
        i:i+window_size]) / window_size
        window_average2 = np.sum(snd_arr[
        i:i+window_size]) / window_size
        
        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)
        moving_averages2.append(window_average2)

        
        # Shift window to right by one position
        i += 1
    return moving_averages, moving_averages2


profitability_arr, profitability_arr2 = moving_avg(samlet_prof, samlet_prof2, window_size)
avg_timend = time.time()
print('ending moving average. time: ', (avg_timend - avgtime))
#np.savetxt("<file>.csv", moving_averages, delimiter = ',')
#print(moving_averages)

end_time = time.time()
print('time:', end_time-start_time)

t_arr1 = np.arange(0,498999)
'''
t_arr2 = np.arange(0,498999)
plt.plot(t_arr1,profitability_arr,'-',label='Firm 1')
plt.plot(t_arr2,profitability_arr2,'-', label='Firm 2')
plt.axhline(y=0.125, color='k', linestyle = '--')
plt.axhline(y=0.061, color='k', linestyle = '--')
plt.xlabel("Time")
plt.ylabel("Profitability")
plt.ylim(0.00,0.15)
plt.legend()
plt.show()
'''

combi_arr = np.mean((np.vstack((profitability_arr, profitability_arr2))), axis=0)

plt.plot(t_arr1,combi_arr,'-',label='Average profit')
plt.axhline(y=0.125, color='k', linestyle = '--')
plt.axhline(y=0.0611, color='k', linestyle = '--')
plt.xlabel("Time")
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
plt.ylabel('Avg. optimality')
plt.ylim(0, 1.5)
plt.legend(loc='lower right')
plt.show()
'''


# Printing prices for each player switching between players

prof_arr, arr, arr1, q_table, bla, bla2, bl3 = game(x, 500000, 0.3, 0.0000276306393827805, 0.95)

print('  Q TABLE 2', q_table)
print('profitability',prof_arr[-10:])
t_arr1 = np.arange(0,499998,2)
t_arr2 = np.arange(1,499999,2)
print('pris1:', arr[-10:])
print('priser2:', arr1[-10:])
print(type(prof_arr))
plt.plot(t_arr1,arr,'--o',label='Player 1', )
plt.plot(t_arr2,arr1,'s--', label='Player 2')
plt.xlabel("Time t")
plt.ylabel("Price")
plt.legend(loc='upper right')
plt.show()



'''
#OPTIMALITY BEREGNING
pro, arr, arr1, Q_t, arr_opt, bla = game( x, 500000, 0.3, 0.0000276306393827805, 0.95)
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


#old many_games, where np.vstack slowed everything down considerably
''''
def many_games(prices, periods, alpha, theta, learners,delta):
    #total_pro_arr = np.zeros((learners,periods-2), dtype=np.ndarray)
    #total_pro_arr2 = np.zeros((learners,periods-2), dtype=np.ndarray)
    #total_opt_arr = np.zeros((learners, 49), dtype=np.ndarray)
    print('run #',1 ,'of ', learners , 'runs') 
    proi_out, arri, arr1i, Q_ti, arr_opt_i_out, proi_out2 = game(prices, periods, alpha, theta, delta)
    for i in range(learners-1):
        print('run #',i+2 ,'of ', learners , 'runs') 
        proi, arri, arr1i, Q_ti, arr_opt_i, proi2 = game(prices, periods, alpha, theta, delta)

        proi_out = np.vstack((proi_out, proi))
        proi_out2 = np.vstack((proi_out2, proi2))
        arr_opt_i_out = np.vstack((arr_opt_i_out, arr_opt_i)) 
        #total_pro_arr2[i] = proi2
        #total_opt_arr[i] = arr_opt_i
        #print('profitability1',proi[-10:])
        #print('profitability1',proi2[-10:])
        #print('pris1:', arri[-10:])
        #print('priser2:', arr1i[-10:])
    return proi_out, arr_opt_i_out, proi_out2
'''
