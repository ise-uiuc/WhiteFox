
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32 * 32, 10)
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(v1.reshape(-1, 32 * 32))
        v3 = torch.tanh(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
