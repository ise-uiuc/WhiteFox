
class Model(torch.nn.Module):
    def __init__(self, size1, size2):
        super().__init__()
        self.size1 = size1
        self.size2 = size2
 
    def forward(self, x1, x2, x3):
        v0 = torch.cat([x1, x2, x3], dim=1)
        v1 = v0[:, 0:9223372036854775807]
        v2 = v1[:, 0:self.size1]
        v3 = torch.zeros([v2.shape[0], v2.shape[1]+v0.shape[1]-self.size1], dtype=torch.float32)
        v3[:, 0:self.size1] = v2
        v4 = v0 + v3
        return v4

# Initializing the model
m = Model(size1=64, size2=32)

# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 32, 32)
x3 = torch.randn(1, 1, 32, 32)
