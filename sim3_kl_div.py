import random
import cvxpy as cp
import csv
from scipy.stats import uniform 
from scipy.stats import truncnorm
import pandas as pd
import matplotlib.pyplot as plt

#normalizes vector to values between 1 and 0 that sum to 1
def final_prob(vec, tot):
    temp_vec = []

    for i in range(0, len(vec)):
        temp_vec.append(vec[i])
        temp_vec[i] = vec[i]/tot
    
    return temp_vec

#Generates player's perceived rewards
def gen_approx_rewards(rewards_vec):
    X_Player = []

    for i in range(0, len(rewards_vec)):
        approx_reward = truncnorm.rvs(a=0, b=10, scale = 1, loc = rewards_vec[i], size=1)
        X_Player.append(approx_reward[0])
    
    return X_Player

#Generates probability vector from rewards vector
def gen_probs(rewards_vec):
    temp_dist = []
    temp_dist = final_prob(rewards_vec, sum(rewards_vec))

    return temp_dist

#P4
def pi_p_signal(bob_alpha, alice_p, bob_q, n):
    pi_p = cp.Variable(n)
    phi = cp.Variable(n)

    alpha_vec = [bob_alpha for i in range(n)]
    alt_vec = [1 - bob_alpha for i in range(n)]

    phi = cp.multiply(alpha_vec, pi_p) + cp.multiply(alt_vec,  bob_q)
    R = cp.kl_div(alice_p, phi)

    objective = cp.Minimize(cp.sum(R))

    constraints = []
    for i in range(0, n):
        constraints += [pi_p[i] >= 0]

    constraints += [sum(pi_p) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    ans = []
    for i in pi_p.value:
        ans.append(i)

    return list(pi_p.value)


def kl_div_iterate(iterations, alpha_val):
    kl_div_list = []
    for iteration in range(0, iterations):
        #arbitrary choices
        choices = [i for i in range (random.randint(2, 20))]
        N = [chr(ord('a')+number) for number in choices]
        n = len(N)

        #Real rewards of those choices
        X = uniform.rvs(0, 10, size=n)
    
        for i in X:
            format(i, 'f')

        X_Alice = gen_approx_rewards(X)
        X_Bob = gen_approx_rewards(X)

        #Alice's prior
        p = gen_probs(X_Alice)

        #Bob's prior
        q = gen_probs(X_Bob)

        pi_p_full = pi_p_signal(alpha_val, p, q, n)
        alice_kl_div = cp.kl_div(pi_p_full, p)
        kl_val = cp.sum(alice_kl_div).value
        kl_div_list.append([f"{kl_val:.11f}"])

    return(kl_div_list)

alpha_vals = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
file_vals = ["low_alpha", "1", "2", "3", "4", "5", "6", "7", "8", "9", "high_alpha"]

for i in range(0, len(alpha_vals)):
    print(alpha_vals[i])
    file_name = "kl_div_" + file_vals[i] + ".csv"

    kl_divs = kl_div_iterate(100, alpha_vals[i])

    with open(file_name, mode='w') as csv_file:
        fieldnames = ['KL_Divergence']
        csvwriter = csv.writer(csv_file)    
        csvwriter.writerow(fieldnames) 
        csvwriter.writerows(kl_divs)

mean_dict = {}

df_low = pd.read_csv('kl_div_low_alpha.csv')
low_mean = df_low.mean(axis = 0)
mean_dict['0.01'] = low_mean[0]

df_1 = pd.read_csv('kl_div_1.csv')
one_mean = df_1.mean(axis = 0)
mean_dict['0.10'] = one_mean[0]

df_2 = pd.read_csv('kl_div_2.csv')
two_mean = df_2.mean(axis = 0)
mean_dict['0.20'] = two_mean[0]

df_3 = pd.read_csv('kl_div_3.csv')
three_mean = df_3.mean(axis = 0)
mean_dict['0.30'] = three_mean[0]

df_4 = pd.read_csv('kl_div_4.csv')
four_mean = df_4.mean(axis = 0)
mean_dict['0.40'] = four_mean[0]

df_5 = pd.read_csv('kl_div_5.csv')
five_mean = df_5.mean(axis = 0)
mean_dict['0.50'] = five_mean[0]

df_6 = pd.read_csv('kl_div_6.csv')
six_mean = df_6.mean(axis = 0)
mean_dict['0.60'] = six_mean[0]

df_7 = pd.read_csv('kl_div_7.csv')
seven_mean = df_7.mean(axis = 0)
mean_dict['0.70'] = seven_mean[0]

df_8 = pd.read_csv('kl_div_8.csv')
eight_mean = df_8.mean(axis = 0)
mean_dict['0.80'] = eight_mean[0]

df_9 = pd.read_csv('kl_div_9.csv')
nine_mean = df_9.mean(axis = 0)
mean_dict['0.90'] = nine_mean[0]

df_high = pd.read_csv('kl_div_high_alpha.csv')
high_mean = df_high.mean(axis = 0)
mean_dict['0.99'] = float(f"{high_mean[0]:.11f}")

print(mean_dict)

keys = list(mean_dict.keys())
values = list(mean_dict.values())

plt.bar(keys, values)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'KL Divergence between $\pi_p(x)$ and $p(x)$')
plt.savefig('kl_div_new.png')