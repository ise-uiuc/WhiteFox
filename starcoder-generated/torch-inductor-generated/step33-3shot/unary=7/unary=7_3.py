
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = torch.tanh(self.conv1(x1))
        v2 = v1 * min(v1, 6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
