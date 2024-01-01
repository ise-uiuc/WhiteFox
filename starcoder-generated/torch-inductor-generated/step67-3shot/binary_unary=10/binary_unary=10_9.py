
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 10)
 
    def forward(self, x):
        v = self.linear(x)
        v2 = v + other
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 512, 1, 1)
