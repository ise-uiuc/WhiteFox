
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(100, 150)
        self.linear2 = torch.nn.Linear(150, 1)
        self.out = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.out(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 100)
