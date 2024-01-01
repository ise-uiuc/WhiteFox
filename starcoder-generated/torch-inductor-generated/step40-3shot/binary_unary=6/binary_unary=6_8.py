
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(8, 10)

    def forward(self, x1):
        v1 = self.linear_layer(x1)
        v2 = v1 - 0.7071067812
        v3 = torch.nn.functional.relu(v1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
