
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(58, 3)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = torch.nn.ReLU()(v2)
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 58)
