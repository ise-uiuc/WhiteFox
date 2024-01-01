
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.linear2 = torch.nn.Linear(20, 30)
        self.linear3 = torch.nn.Linear(30, 10)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = clamp(0, 6, v1 + 3)
        v3 = v2 / 6
        v4 = self.linear2(v3)
        v5 = torch.clamp(-1, 1., v4)
        v6 = self.linear3(v5)
        return v6

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(5, 10)
