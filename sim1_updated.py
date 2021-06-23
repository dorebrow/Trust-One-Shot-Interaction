import random
import cvxpy as cp
import string


#normalizes vector to values between 1 and 0 that sum to 1
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
def pi_p_signal(bob_alpha, alice_p, bob_q):
    print("\nSending pi_p:")
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

    print("sum of pi_p value: ", sum(pi_p).value)

    print('Problem status: ', prob.status)
    print("Value of pi_p:", pi_p.value)

    print("phi: ", phi.value)
    print("Sum of pi_p: ", sum(pi_p.value))

    ans = []
    for i in pi_p.value:
        ans.append(i)
    print("pi_p: ", ans)

    return list(pi_p.value)

#P3
def exp_pi_p_signal(bob_alpha, alice_p, bob_q, rewards):
    print("\nSending just expectation of pi_p:")
    pi_p = cp.Variable(n)

    exp_pi_p = cp.multiply(pi_p, rewards)

    R = (cp.multiply(alpha, exp_pi_p) + cp.multiply((1 - alpha), exp(bob_q, rewards)) - exp(p, rewards))**2

    objective = cp.Minimize(cp.sum(R))

    constraints = []
    for i in range(0, n):
        constraints += [pi_p[i] >= 0]

    constraints += [sum(pi_p) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    print('Problem status: ', prob.status)
    print("Value of Exp pi_p:", exp(pi_p.value, rewards))

    print("Sum of pi_p: ", sum(pi_p.value))

    ans = []
    for i in pi_p.value:
        ans.append(i)
    print("pi_p: ", ans)

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
def bob_independent_opt_choice(bob_q, rewards):
    q_exp_vec = exp_vec(bob_q, rewards)
    bob_psi = psi(q_exp_vec)
    choice_index = bob_choice(bob_psi)
    bob_util = q_exp_vec[choice_index]


    return choice_index, bob_util

def gen_approx_rewards(X, sig):
    X_Alice = []

    for i in range(0, len(X)):
        X_Alice.append(random.gauss(mu=X[i], sigma=sig))
    
    return X_Alice

def gen_probs(X):
    temp_dist = []
    temp_dist = final_prob(X, sum(X))

    return temp_dist


#arbitrary choices
choices = [i for i in range (random.randint(2, 10))]
N = [chr(ord('a')+number) for number in range(0, len(choices))]

n = len(N)
print("Choices: ", N)

#Real rewards of those choices
X = [random.gauss(mu=5, sigma=1) for n in N]
true_dist = gen_probs(X)

print("Rewards: ", X)
print("True dist: ", true_dist, "\n")

X_Alice = gen_approx_rewards(X, 1)
print("Alice rewards: ", X_Alice)
X_Bob = gen_approx_rewards(X, 1)
print("Bob rewards: ", X_Bob, '\n')

#Alice's prior
p = gen_probs(X_Alice)

#Bob's prior
q = gen_probs(X_Bob)

alpha = 0.5

print("alpha: ", alpha)
print("p: ", p)
print("q: ", q)

pi_p_full = pi_p_signal(alpha, p, q)

pi_p_exp = exp_pi_p_signal(alpha, p, q, X)

exp_p = exp_vec(p, X)

y1 = exp_phi_vec(X, alpha, p, q, pi_p_full)
bob_psi1 = psi(y1)
choice_ind1 = bob_choice(bob_psi1)
bob_util1 = util(bob_psi1, y1, choice_ind1)
alice_util1 = util(bob_psi1, exp_p, choice_ind1)

print("\nPhi when Alice sends full distribution pi_p: ")
print("Expectation of phi vector: ", y1)
print("Psi at Bob: ", bob_psi1)
print("Bob's choice: ", choice_ind1, "with label", N[choice_ind1])
print("Bob's utility: ", bob_util1)
print("Alice's utility: ", alice_util1)

y2 = exp_phi_vec(X, alpha, p, q, pi_p_exp)
bob_psi2 = psi(y2)
choice_ind2 = bob_choice(bob_psi2)
bob_util2 = util(bob_psi2, y2, choice_ind2)
alice_util2 = util(bob_psi2, exp_p, choice_ind2)

print("\nPhi when Alice sends expectation of pi_p: ")
print("Expectation of phi vector: ", y2)
print("Psi at Bob: ", bob_psi2)
print("Bob's choice: ", choice_ind2, "with label", N[choice_ind2])
print("Bob's utility: ", bob_util2)
print("Alice's utility: ", alice_util2)

bob_ind_index, bob_ind_util = bob_independent_opt_choice(q, X)
print("\nBob's best choice independent of Alice: ", bob_ind_index, "with label", N[bob_ind_index], "and utility", bob_ind_util)
print("\nQuality of Alice's recommendation when she sends full pi_p: ", bob_ind_util - bob_util1)
print("Quality of Alice's recommendation when she sends exp pi_p: ", bob_ind_util - bob_util2)