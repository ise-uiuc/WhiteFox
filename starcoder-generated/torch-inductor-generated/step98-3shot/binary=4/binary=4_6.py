
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 16)
 
    def forward(self, x, other):
        x = self.linear(x)
        return x + other

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 32)
other = torch.randn(2, 16)
