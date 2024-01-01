
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = x1.flatten(0, 1)
        v2 = x1 + 5
        v3 = torch.nn.functional.relu(v2)
        return v3 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
