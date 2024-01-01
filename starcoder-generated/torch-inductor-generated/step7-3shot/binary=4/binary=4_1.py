
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(32, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 10)
 
    def forward(self, x):
        e1 = self.linear1(x)
        e2 = self.linear2(e1)
        e3 = e2 + self.linear3(e1)
        return e3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 32)
