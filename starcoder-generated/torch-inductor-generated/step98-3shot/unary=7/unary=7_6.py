
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * math.tanh(0.5 * torch.minimum(6, torch.maximum(0, v1 + 3))) 
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32) 
