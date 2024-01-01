
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = torch.clamp_min(v1, 0.5)
        v3 = torch.clamp_max(v2, -0.5)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
