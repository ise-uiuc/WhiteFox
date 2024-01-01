
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28 * 28, 10)
 
    def forward(self, x):
        v1 = self.linear(x)
        return torch.clamp_max(torch.clamp_min(v1, min=-15.), max=15.)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.rand(1, 28 * 28)
