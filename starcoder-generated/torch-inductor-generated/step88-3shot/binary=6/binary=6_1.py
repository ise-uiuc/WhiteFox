
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 32, bias=True)
        self.linear2 = torch.nn.Linear(32, 16, bias=True)
        self.linear3 = torch.nn.Linear(16, 8, bias=True)
 
    def forward(self, x1):
        v1 = self.linear1(x1).tanh()
        v2 = self.linear2(v1).tanh()
        v3 = self.linear3(v2).tanh()
        v4 = v1 - v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
