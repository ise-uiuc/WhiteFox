
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(20, 10)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 0.20
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(20)
