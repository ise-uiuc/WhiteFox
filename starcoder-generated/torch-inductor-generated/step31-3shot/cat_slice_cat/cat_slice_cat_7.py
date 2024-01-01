
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1):
        v1 = torch.cat(input_tensors, dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model(size=5)

# Inputs to the model
x1 = torch.randn(1, 10, 512, 512)
x2 = torch.randn(1, 10, 512, 512)
