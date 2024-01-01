
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
 
    def forward(self, x1, x2):
        w1 = torch.cat([x1, x2], dim=1)
        w2 = w1[:, 0:9223372036854775807]
        w3 = w2[:, 0:self.size]
        w4 = torch.cat([w1, w3], dim=1)
        return w4

# Initializing the model
m = Model(size=size)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
