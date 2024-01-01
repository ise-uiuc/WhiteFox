
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 3)
        self.linear2 = torch.nn.Linear(5, 3)
 
    def forward(self, x1, x2):
        v1 = self.linear1(x1) + 1
        v2 = self.linear2(x2)
        return torch.tanh(v1 + v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 5)
