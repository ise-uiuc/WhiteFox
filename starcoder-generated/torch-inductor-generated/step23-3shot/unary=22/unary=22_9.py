
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 1)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.tanh(v1)
        y = self.linear2(v2)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
