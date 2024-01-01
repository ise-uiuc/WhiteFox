
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 100)
        v3 = torch.clamp_max(v2, 200)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
