
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 0.00015544818718936095)
        return torch.clamp_max(v2, 0.0008010327622311562)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
