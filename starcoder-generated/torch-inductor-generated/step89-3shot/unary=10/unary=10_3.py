
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1.clamp_min(0).clamp_max(6) / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, )
