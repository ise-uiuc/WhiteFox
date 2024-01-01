
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 1)
        self.linear2 = torch.nn.Linear(3, 1)
        self.linear3 = torch.nn.Linear(1, 1)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1).relu()
        v2 = self.linear2(x2).tanh()
        v3 = self.linear3(v1 * v2).sigmoid()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
