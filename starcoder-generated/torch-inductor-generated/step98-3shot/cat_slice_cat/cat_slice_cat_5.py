
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # Slice an output tensor t1 along dimension 0 to size 40
    def slice_tensor(self, t1, size):
        return t1[:, 0:size]
    
    def forward(self, x1):
        v1 = torch.cat(x1, dim=1)
        v2 = self.slice_tensor(v1, torch.iinfo(torch.int64).max)
        v3 = self.slice_tensor(v1, 40)
        return torch.cat([v1, v3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 64, 256, 1)
x2 = torch.randn(4, 256, 256, 1)
x3 = torch.randn(4, 128, 256, 1)
x4 = torch.randn(4, 16, 256, 1)
