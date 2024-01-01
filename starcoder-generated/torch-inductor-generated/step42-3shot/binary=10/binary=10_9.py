
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 10)
 
    def forward(self, x1, param=None):
        v1 = self.linear(x1)
        v2 = v1 + param
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.Tensor(np.random.random((10, 13)))
param = torch.Tensor(np.random.random((10, 10)))
