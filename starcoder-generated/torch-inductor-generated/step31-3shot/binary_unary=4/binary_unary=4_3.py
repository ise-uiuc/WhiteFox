
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(1000, 512)
        self.bias = torch.randn(512)
 
    def forward(self, x1, other=None):
        v1 = torch.matmul(x1, self.weight.t())
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000)
other = torch.randn(1, 512)
