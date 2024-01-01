
class Model():
    def __init__(self):
        pass
 
    def forward(self, x, other):
        v1 = torch.nn.functional.linear(x, 16)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
other = torch.randn(1, 16)
