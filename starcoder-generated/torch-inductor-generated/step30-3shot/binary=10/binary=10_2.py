
def m(x):
    v1 = x.reshape(-1, x.shape[-1])
    return v1.T

# Initializing the model
__output = m(x)

