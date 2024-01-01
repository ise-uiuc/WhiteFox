
def func(f0, f1, f2, x4):
    v2 = f0(x4)
    v3 = x4 * f1(x4)
    v5 = x4 - f0(x4)
    v6 = f2(x4)
    v7 = f2(x4)
    v10 = f2(v2)
    v11 = v5 * v6
    v12 = v5 - v10
    v15 = v6 * v7
    v16 = v12 + v15
    v17 = v11 * v16
    return v17

# Initializing the model
