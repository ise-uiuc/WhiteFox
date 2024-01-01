
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, __input_tensor__, __other__ = None):
        v1 = self.linear(__input_tensor__)
        v2 = v1 + __other__
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
o1 = torch.randn(1, 256)
