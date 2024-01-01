
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 1)
 
    def forward(self, x):
        x2 = self.linear(x)
        o1 = torch.sigmoid(x2)
        return o1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 256)
