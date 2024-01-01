
class Model(torch.nn.Module):
    def __init__(self, size, offset, size1, offset1):
        super().__init__()
        self.size = size
        self.offset = offset
        self.size1 = size1
        self.offset1 = offset1
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, self.offset:self.size]
        v3 = v2[:, self.offset1:self.size1]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model(9223372036854775807, 0, 9223372036854775807, 0)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
