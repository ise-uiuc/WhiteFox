
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 10)
 
    def forward(self, x2):
        c1 = self.linear(x2)
        r1 = torch.relu(c1)
        return r1

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
