
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 50)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_max(torch.clamp(v1, 0), 50)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
