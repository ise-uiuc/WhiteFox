
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = torch.nn.Linear(3, 2)
 
    def forward(self, x1):
        v1 = self._linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
