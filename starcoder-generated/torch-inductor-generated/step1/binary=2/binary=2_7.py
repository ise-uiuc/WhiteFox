
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        return v1 - other

# Initializing the model
m = Model()

def func(x):
    return x * 7
m.register_buffer("other", torch.tensor(1.0,requires_grad=True))
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
