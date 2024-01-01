
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp =  torch.nn.Linear(16, 32)
        self.linear3 =  torch.nn.Linear(32, 32)
 
    def forward(self, x1, other):
        x2 = self.mlp(x1)
        x3 = self.linear3(x2)
        x4 = x3 + other
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8, 16)
other = torch.randn(8, 32)
