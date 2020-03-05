import numpy as np
import cvxpy as cvx
import random

random.seed(0)
dim = 50
alpha = 0.5
max_iters = 50

def bsum_main(objective, x, y, block, op_var="x", x_init=None, y_init=None,):

    if op_var == "x":
        constraints = [y == y_init]
        for i in range(len(x_init)):
            if i not in block:
                constraints.append(x[i] == x_init[i])
    else:
        constraints = [x == x_init]
        for i in range(len(y_init)):
            if i not in block:
                constraints.append(y[i] == y_init[i])

    problem = cvx.Problem(objective=objective, constraints=constraints)
    problem.solve("CVXOPT")
    for i in block:
        if op_var == "x":
            x_init[i] = x.value[i]
        else:
            y_init[i] = y.value[i]

    return x, y, problem.objective.args[0].value

x_init = np.zeros((dim, 1), dtype=np.float32)
y_init = np.random.randn(dim, 1)
x = cvx.Variable(dim)
y = cvx.Variable(dim)
x.value = x_init
y.value = y_init
#         minimize    1/2 ||y||^2
#               s.t.    Ax - b = y
ObjectiveFn = (1/(2*alpha)) * cvx.square(cvx.norm(x-y)) +  0.5 * cvx.norm ( x )**2
objective = cvx.Minimize(ObjectiveFn)

block_size = 10
no_of_blocks = int(dim / block_size)
blocks = []
for i in range(no_of_blocks):
    blocks.append([j for j in range(i*block_size, (i*block_size+block_size))])

cyclicObj = []
randomObj = []

for it in range(max_iters):
    for block in blocks:
        x, y, cur_obj = bsum_main(objective, x, y, block, op_var="x", x_init=x_init, y_init=y_init)
    cyclicObj.append(cur_obj)
    print("cyclic iter: " + str(it) + ", obejctive: " + str(cur_obj))
    for j in range(5):
        r = random.randint(0, 4)
        block = blocks[r]
        x, y, cur_obj = bsum_main(objective, x, y, block, op_var="x", x_init=x_init, y_init=y_init)
    randomObj.append(cur_obj)
    print("random iter: " + str(it) + ", obejctive: " + str(cur_obj))

print "cyclicObj = ",cyclicObj
print "randomObj = ",randomObj

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(range(1, max_iters + 1), cyclicObj, label='Cyclic coordinate descent', linewidth=5.0)
ax.plot(range(1, max_iters + 1), randomObj, c='red', label='Randomized coordinate descent')
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('Objective value vs. iteration ')
ax.legend(loc=0)
plt.show()




