
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 128)
 
    def forward(self, x):
        v1 = self.linear(x)
        return torch.clamp_min(torch.clamp_max(v1, min=0), max=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,4)
min_value = -1
max_value = 1
