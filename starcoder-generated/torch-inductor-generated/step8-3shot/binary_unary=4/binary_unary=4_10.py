
class Model(torch.nn.Module):
    def __init__(self, other=None):
        super().__init__()
        self.linear = torch.nn.Linear(13, 42)
    
    def forward(self, x1):
        v1 = self.linear(x1)
        return {'output': v1 + other}

# Initializing the model
m = Model(other=torch.randn(1, 42))

# Inputs to the model
x1 = torch.randn(1, 13)
