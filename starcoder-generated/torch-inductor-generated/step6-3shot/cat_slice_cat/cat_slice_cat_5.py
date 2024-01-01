
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slice_dim1 = slice(0, torch.iinfo(torch.int64).max // 2)
        self.slice_dim2 = slice(0, torch.iinfo(torch.int64).max)
        self.dim1 = torch.iinfo(torch.int64).max // 2
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, self.slice_dim1]
        v3 = v2[:, self.slice_dim2]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Dimensions of the input tensors
x1 = torch.randn(2, 64, 64, 64, 64)
x2 = torch.randn(2, 32, 32, 32, 32)
