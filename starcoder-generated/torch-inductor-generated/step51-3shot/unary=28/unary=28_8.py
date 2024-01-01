
class Model(torch.nn.Module):
    def __init__(self, min_value=-10):
        super().__init__()
        self.linear = torch.nn.Linear(784, 10, bias=True)
 
    def forward(self, x1):
        x11 = torch.reshape(x1, (-1, 784))
        v1 = self.linear(x11)
        v2 = torch.clamp_min(v1, min_value=min_value)
        v3 = torch.clamp_max(v2, max_value=10.5)
        return torch.argmax(v3, dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 28, 28)
