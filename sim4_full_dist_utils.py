import random
import cvxpy as cp
import csv
from scipy.stats import uniform 
from scipy.stats import truncnorm

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


#need some large number of iterations
def full_dist_util_iterate(iterations, num_choices):
    average_util_alice = 0
    average_util_bob = 0
    total_util_alice = 0
    total_util_bob = 0
    
    for i in range(iterations):
        choices = [i for i in range(num_choices)]
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

        alpha = 0.5

        pi_p_full = pi_p_signal(alpha, p, q, n)

        exp_p = exp_vec(p, X_Alice)
        
        y1 = exp_phi_vec(X_Bob, alpha, p, q, pi_p_full)
        bob_psi1 = psi(y1)
        choice_ind1 = bob_choice(bob_psi1)
        bob_util1 = util(bob_psi1, y1, choice_ind1)

        #look at alice util
        alice_util1 = util(bob_psi1, exp_p, choice_ind1)

        total_util_alice += alice_util1
        total_util_bob += bob_util1
    
    average_util_alice = total_util_alice / iterations
    average_util_bob = total_util_bob / iterations

    return average_util_alice, average_util_bob

util_dict_bob = {}
util_dict_alice = {}

for i in range(2, 21):
    print(i)

    alice_util, bob_util = full_dist_util_iterate(1000, i)

    util_dict_alice[str(i)] = alice_util
    util_dict_bob[str(i)] = bob_util

print("Alice Utilities:", util_dict_alice)
print("Bob Utilities:", util_dict_bob)