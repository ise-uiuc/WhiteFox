
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 7)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 0.5)
        v3 = torch.clamp_max(v2, 0.7071067811865476)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 13)
