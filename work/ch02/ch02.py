#%% [markdown]
# 2.3 パーセプトロンの実装

#%%
def AND0(x1, x2):
    w1, w2, theta = 0.5, 0.5, 1
    if w1*x1 + w2*x2 >= theta:
        return 1
    else:
        return 0

#%%
print( "AND(0, 0)", AND(0, 0))
print( "AND(1, 0)", AND(1, 0))
print( "AND(0, 1)", AND(0, 1))
print( "AND(1, 1)", AND(1, 1))


#%%
import numpy as np
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
w*x

#%%
np.sum(w*x)+b

#%%
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    if np.sum(w*x)+b >= 0:
        return 1
    else:
        return 0

#%%
print( "AND(0, 0)", AND(0, 0))
print( "AND(1, 0)", AND(1, 0))
print( "AND(0, 1)", AND(0, 1))
print( "AND(1, 1)", AND(1, 1))

#%%
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    if np.sum(w*x)+b >= 0:
        return 1
    else:
        return 0

#%%
print( "NAND(0, 0)", NAND(0, 0))
print( "NAND(1, 0)", NAND(1, 0))
print( "NAND(0, 1)", NAND(0, 1))
print( "NAND(1, 1)", NAND(1, 1))

#%%
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = -1
    if np.sum(w*x)+b >= 0:
        return 1
    else:
        return 0

#%%
print( "OR(0, 0)", OR(0, 0))
print( "OR(1, 0)", OR(1, 0))
print( "OR(0, 1)", OR(0, 1))
print( "OR(1, 1)", OR(1, 1))

#%% [markdown]
# パーセプトロンの限界

#%%
def XOR(x1, x2):
    a = OR(x1, x2)
    b = NAND(x1, x2)
    if AND(a, b) > 0:
        return 1
    else:
        return 0

#%%
print( "XOR(0, 0)", XOR(0, 0))
print( "XOR(1, 0)", XOR(1, 0))
print( "XOR(0, 1)", XOR(0, 1))
print( "XOR(1, 1)", XOR(1, 1))
