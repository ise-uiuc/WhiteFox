
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
input_tensors = [1, 2, 3, 4, 5]
size = 2
m = Model(size)

# Inputs to the model
x1 = torch.randn(1, 1, 2, 3, 4)
x2 = torch.randn(1, 2, 3, 4)
x3 = torch.randn(1, 3, 2, 3)
x4 = torch.randn(1, 4, 3)
x5 = torch.randn(1, 5)
