
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(2, 16, 64, 64)
