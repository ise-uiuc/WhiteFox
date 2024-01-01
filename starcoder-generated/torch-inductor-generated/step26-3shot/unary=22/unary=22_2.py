
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 64)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.reshape(1, -1)
        v3 = torch.tanh(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
