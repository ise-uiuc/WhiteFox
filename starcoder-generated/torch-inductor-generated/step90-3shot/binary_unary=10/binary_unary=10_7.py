
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 2.562631692347981
        v3 = torch.relu(v2)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
