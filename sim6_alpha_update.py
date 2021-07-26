import random
import cvxpy as cp
import numpy as np
from scipy.stats import uniform 
from scipy.stats import truncnorm
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

def final_prob(vec, tot):
    temp_vec = []

    for i in range(0, len(vec)):
        temp_vec.append(vec[i])
        temp_vec[i] = vec[i]/tot
    
    return temp_vec

#returns summed expected value vector
def exp(dist, rewards):
    exp = []
    for i in range(len(rewards)):
        exp.append(dist[i] * rewards[i])
    return sum(exp)

#returns vector of expected values
def exp_vec(dist, rewards):
    exp = []
    for i in range(len(rewards)):
        exp.append(dist[i] * rewards[i])
    return exp

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

#P3
def exp_pi_p_signal(alpha, alice_p, bob_q, rewards, n):
    pi_p = cp.Variable(n)

    exp_pi_p = cp.multiply(pi_p, rewards)

    R = (cp.multiply(alpha, exp_pi_p) + cp.multiply((1 - alpha), exp(bob_q, rewards)) - exp(alice_p, rewards))**2

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

#returns y vector
def exp_phi_vec(rewards, bob_alpha, alice_p, bob_q, pi_p_vec):
    y = []

    pi_p_exp_vec = exp_vec(pi_p_vec, rewards)
    q_exp_vec = exp_vec(bob_q, rewards)

    for i in range(len(rewards)):
        y.append((bob_alpha * pi_p_exp_vec[i]) + (1 - bob_alpha) * q_exp_vec[i])

    return y

#returns psi vector
def psi(exp_phi_vector):
    psi_vec = [0] * len(exp_phi_vector)

    max_index =  max(range(len(exp_phi_vector)), key = lambda x: exp_phi_vector[x])
    psi_vec[max_index] = 1

    return psi_vec

#determines Bob's choice index based on psi
def bob_choice(psi_vector):
    for i in psi_vector:
        if i == 1:
            return psi_vector.index(i)

#returns utility
def util(psi_vector, y, choice_index):
    return psi_vector[choice_index] * y[choice_index]

#returns the choice Bob would have made without Alice's input
def bob_independent_opt_choice(bob_q, player_rewards_vec, y):
    q_exp_vec = exp_vec(bob_q, player_rewards_vec)
    bob_psi = psi(q_exp_vec)
    choice_index = bob_choice(bob_psi)
    bob_util = util(bob_psi, y, choice_index)

    return choice_index, bob_util


#Generates player's perceived rewards
def gen_approx_rewards(rewards_vec):
    mu, sigma = 0, 1

    X_Player = []

    for i in range(0, len(rewards_vec)):
        #approx_reward = truncnorm.rvs(a=0, b=10, scale = 1, loc = rewards_vec[i], size=1)
        s = np.random.normal(mu, sigma, 1)
        approx_reward =  rewards_vec[i] + s[0]

        while rewards_vec[i] + s[0] < 0:
            s = np.random.normal(mu, sigma, 1)
            approx_reward =  rewards_vec[i] + s[0]

        X_Player.append(approx_reward)
        
    return X_Player



#Generates probability vector from rewards vector
def gen_probs(rewards_vec):
    temp_dist = []
    temp_dist = final_prob(rewards_vec, sum(rewards_vec))

    return temp_dist

#Determines regret at Bob for each signal
def regret_comparison_iterate(iterations, num_choices, alpha_val, epsilon):
    bob_average_1 = 0
    bob_average_2 = 0
    bob_total_1 = 0
    bob_total_2 = 0
    alpha_total_1 = 0
    alpha_total_2 = 0

    for i in range(iterations):
        choices = [i for i in range(num_choices)]
        N = [chr(ord('a')+number) for number in choices]
        n = len(N)

        #Real rewards of those choices
        X = uniform.rvs(0, 10, size=n)
    
        for i in X:
            format(i, 'f')
        
        true_dist = gen_probs(X)
        
        X_Alice = gen_approx_rewards(X)
        X_Bob = gen_approx_rewards(X)

        #Alice's prior
        p = gen_probs(X_Alice)

        #Bob's prior
        q = gen_probs(X_Bob)

        alpha = alpha_val

        pi_p_full = pi_p_signal(alpha, p, q, n)

        #exp_pi_p_signal solves P3 at Alice, so it should be solved using Alice's perceived rewards
        pi_p_exp = exp_pi_p_signal(alpha, p, q, X_Alice, n)

        y1 = exp_phi_vec(X_Bob, alpha, p, q, pi_p_full)
        bob_psi1 = psi(y1)
        choice_ind1 = bob_choice(bob_psi1)
        bob_util1 = util(bob_psi1, y1, choice_ind1)

        y2 = exp_phi_vec(X_Bob, alpha, p, q, pi_p_exp)
        bob_psi2 = psi(y2)
        choice_ind2 = bob_choice(bob_psi2)
        bob_util2 = util(bob_psi2, y2, choice_ind2)

        #Captures Bob's regret
        y3 = exp_phi_vec(X_Bob, 0, p, q, pi_p_full)
        bob_ind_index, bob_ind_util = bob_independent_opt_choice(q, X_Bob, y3)

        #Bob's regret when Alice sends full pi_p
        bob_regret_1 = bob_ind_util - bob_util1

        #Bob's regret when Alice sends exp pi_p
        bob_regret_2 = bob_ind_util - bob_util2

        alpha_prime_1 = update_alpha(alpha, bob_regret_1, epsilon)
        alpha_prime_2 = update_alpha(alpha, bob_regret_2, epsilon)

        bob_total_1 += bob_regret_1
        bob_total_2 += bob_regret_2

        alpha_total_1 += alpha_prime_1
        alpha_total_2 += alpha_prime_2

    bob_average_1 = bob_total_1 / iterations
    bob_average_2 = bob_total_2 / iterations
    
    alpha_update_average_1 = alpha_total_1 / iterations
    alpha_update_average_2 = alpha_total_2 / iterations

    return bob_average_1, bob_average_2, alpha_update_average_1, alpha_update_average_2


def generate_regret_trust_dicts(alpha_list, epsilon):
    bob_regret_full = {}
    bob_regret_exp = {}
    bob_alpha_full = {}
    bob_alpha_exp = {}

    for val in alpha_list:
        print(val)
        bob_avg_reg_full, bob_avg_reg_exp, bob_avg_alpha_full, bob_avg_alpha_exp = regret_comparison_iterate(500, 10, val, epsilon)
        bob_regret_full[str(val)] = bob_avg_reg_full
        bob_regret_exp[str(val)] = bob_avg_reg_exp
        bob_alpha_full[str(val)] = bob_avg_alpha_full
        bob_alpha_exp[str(val)] = bob_avg_alpha_exp

    print("Bob reg full: ", bob_regret_full)
    print("Bob reg exp: ", bob_regret_exp)

    return bob_regret_full, bob_regret_exp, bob_alpha_full, bob_alpha_exp


#Bob's update rule for alpha
def update_alpha(alpha, regret, epsilon):
    reg_list = [regret]
    reg_sign = np.sign(reg_list)[0]

    alpha_prime = alpha - (epsilon * reg_sign)

    if alpha_prime > 1:
        alpha_prime = 1
    elif alpha_prime < 0:
        alpha_prime = 0
    else:
        return alpha_prime

    return alpha_prime

alpha_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
epsilon = 0.5

bob_reg_full, bob_reg_exp, updated_alphas_full, updated_alphas_exp = generate_regret_trust_dicts(alpha_list, epsilon)

print("New alpha vals for full dist: ", updated_alphas_full)
print("New alpha vals for exp", updated_alphas_exp)

fig, ax = plt.subplots()
ax.plot(list(updated_alphas_full.keys()),list(updated_alphas_full.values()), color = '#1f77b4', label=r'$\pi_p(x)$', marker = 'o')
ax.plot(list(updated_alphas_exp.keys()), list(updated_alphas_exp.values()), color = 'orange', label=r'$\mathbb{E}_{\pi_p}(x)$', marker = 'o')
ax.set_xlabel(r'$\alpha$', fontsize=16)  # Add an x-label to the axes.
ax.set_ylabel(r'$\alpha^\prime$', fontsize=16)  # Add a y-label to the axes.
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(prop={"size":14})
plt.show()
