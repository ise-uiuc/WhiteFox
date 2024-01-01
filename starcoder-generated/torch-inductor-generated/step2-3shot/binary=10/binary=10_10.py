
class Model(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.other = tensor
 
    def forward(self, x1):
        v1 = x1.mean()
        v2 = self.other.mean()
        v3 = x1 * v2
        v3 = v3 / 10 + v1
        v4 = v3[v3 < 0]
        return v4

# Initializing the model
m = Model(torch.randn(10))

# Inputs to the model
x1 = torch.randn(20, 10)
