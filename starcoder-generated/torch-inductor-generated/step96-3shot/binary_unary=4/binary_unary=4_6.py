
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1, a):
        v1 = self.conv(x1)
        v2 = self.linear(v1)
        v3 = v2 + a
        return self.relu(v3)
 
def relu(v):
    # v is the input tensor
    return F.relu(F.max_pool2d(v, 2))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
a = torch.zeros(1, 16)
