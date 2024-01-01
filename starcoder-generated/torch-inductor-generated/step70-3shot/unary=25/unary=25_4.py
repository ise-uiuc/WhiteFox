
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 > 0
        v3 = v1 * 0.01
        return torch.where(v2, v1, v3)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
