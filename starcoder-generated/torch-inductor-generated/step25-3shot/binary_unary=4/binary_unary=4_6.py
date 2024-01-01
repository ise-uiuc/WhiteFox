
def linear(x, weight, bias):
    linear = torch.nn.functional.linear(x, weight, bias)
    return linear

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 8)
 
    def forward(self, x1, x2 = None):
        v1 = self.conv(x1)
        if x2 is None:
            x2 = torch.randn(1, 5, 64, 64)
        v2 = linear(v1, x2)
        v3 = torch.relu(v1)
        return v3

# Initializing the model
m = Model()

