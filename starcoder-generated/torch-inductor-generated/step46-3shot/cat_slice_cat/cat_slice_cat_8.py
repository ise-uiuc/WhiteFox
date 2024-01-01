
class Model(torch.nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
 
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = torch.cat([x1,x2], dim=1)
        v2 = v1[:, 0:self.dim1]
        v3 = v2[:, 0:self.dim2]
        v4 = torch.cat([x3,x4,x5,x6], dim=1)
        return v4
 
# Initializing the model
m = Model(dim1, dim2)
 
# Inputs to the model
x1 = torch.randn(batch_size, dim0, dim1)
x2 = torch.randn(batch_size, dim0, dim1)
x3 = torch.randn(batch_size, dim0, dim1)
x4 = torch.randn(batch_size, dim0, dim1)
x5 = torch.randn(batch_size, dim0, dim1)
x6 = torch.randn(batch_size, dim0, dim1)

