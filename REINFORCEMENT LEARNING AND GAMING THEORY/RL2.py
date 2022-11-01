map = []
q = [[]]
import random

import numpy as np

action = [(-1,0),(0,1),(1,0),(0,-1)]
def get_reward(i,j,action_num):
    a = action[action_num]
    if i + a[0] < 0 or i + a[0] >= len(map) or j + a[1] < 0 or j + a[1] >= len(map[0]):
        return (i,j), -1
    if map[i + a[0]][j + a[1]] == 1:
        return (i,j), -100
    return (i + a[0],j + a[1]),-1
# class q(object):
#     def __init__(self,map) -> None:
#         self.q = np.zeros((len(map),len(map[0]),4))
    

        

map = [[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,1,1,1,1,0]]


# alpha for exploration
def sarasa_derive_action(q,s,alpha):
    action_record = np.argmax(q[s[0]][s[1]])
    rate = [alpha / 4,alpha / 4,alpha / 4,alpha / 4]
    rate[action_record] = alpha / 4 + 1 - alpha
    r = random.random()
    for i in range(4):
        r -= rate[i]
        if r < 0:
            return i
    return -1
def sarsa(st,ed,alpha,learning_rate,discount): 
    q = np.zeros((len(map),len(map[0]),4))
    bl = 1

    cnt = 0
    while bl:
        s = st
        a = sarasa_derive_action(q,s,alpha)
        while s != ed:
            s_next,r = get_reward(s[0],s[1],a)
            a_next = sarasa_derive_action(q,s_next,alpha)
            q[s[0]][s[1]][a] = q[s[0]][s[1]][a] * (1 - learning_rate) + learning_rate * (r + discount * q[s_next[0]][s_next[1]][a_next])
            s = s_next
            a = a_next
        cnt += 1
        if cnt == 1000:
            bl = 0
    return q
def q_learning(st,ed,alpha,learning_rate,discount):
    q = np.zeros((len(map),len(map[0]),4))
    bl = 1
    cnt = 0
    while bl:
        s = st
        while s != ed:
            a = sarasa_derive_action(q,s,alpha)
            s_next,r = get_reward(s[0],s[1],a)
            qs_nextamax = max(q[s_next[0]][s_next[1]])
            q[s[0]][s[1]][a] = q[s[0]][s[1]][a] * (1 - learning_rate) + learning_rate *(r + discount * qs_nextamax)
            s = s_next
        cnt += 1
        if cnt == 1000:
            bl = 0
    return q

def getting_route_from_a_policy(q,st,ed):
    s = st
    rwd = 0
    while s != ed:
        print(s)
        a =  np.argmax(q[s[0]][s[1]])
        s_next,r = get_reward(s[0],s[1],a)
        s = s_next
        

print("SARSA")
q = sarsa((3,0),(3, 11),0.05,0.05,1)
getting_route_from_a_policy(q,(3,0),(3,11))
print("Q-learning")
q = q_learning((3,0),(3, 11),0.1,0.05,1)
getting_route_from_a_policy(q,(3,0),(3,11))
