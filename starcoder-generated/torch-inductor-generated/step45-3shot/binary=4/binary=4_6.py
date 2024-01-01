
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(20, 4)
        self.linear2 = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        v1 = torch.tanh(self.linear1(x1))
        v2 = v1 + self.linear2(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
