
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 9)
 
    def func(self, x):
        v2 = self.linear(x)
        v3 = torch.reshape(v2, (6, 3))
        v3 = v3 * 12
        v4 = torch.relu(v3)
        v4 = v4 * 3
        v5 = torch.softmax(v4, dim=-1)
        v6 = v5 * 0.5
        v7 = v5 * 1.4142135623730951
        v8 = torch.atan(v7)
        v8 = v8 * 6
        t1 = torch.cat([v8, v6, v7], dim=0)
        return t1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3)
