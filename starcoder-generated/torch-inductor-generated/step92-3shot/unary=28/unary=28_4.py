
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 20)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = torch.clamp_min(v2, min=v2.min())
        v4 = torch.clamp_max(v3, max=v3.max())
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 1)
max_value = x2.max()
min_value = x2.min()
