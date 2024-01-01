
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x1, o1):
        v1 = self.linear(x1)
        v2 = v1 + o1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(100, 128)
x2 = torch.randn(256, 128)
o1 = torch.randn(256)
