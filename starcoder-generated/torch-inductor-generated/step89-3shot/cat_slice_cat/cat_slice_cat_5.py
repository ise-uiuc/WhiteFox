
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17):
        v2 = torch.cat((x1, x2, x3), 1)
        v5 = torch.cat((v2, x4), 1)
        v9 = torch.cat((v5, x5), 1)
        v13 = torch.cat((v9, x6), 1)
        v17 = torch.cat((v13, x7), 1)
        v21 = torch.cat((v17, x8), 1)
        v25 = torch.cat((v21, x9), 1)
        v29 = torch.cat((v25, x10), 1)
        v33 = torch.cat((v29, x11), 1)
        v37 = torch.cat((v33, x12), 1)
        v41 = torch.cat((v37, x13), 1)
        v45 = torch.cat((v41, x14), 1)
        v49 = torch.cat((v45, x15), 1)
        v53 = torch.cat((v49, x16), 1)
        v57 = torch.cat((v53, x17), 1)
        return v57

# Initializing the model
m = Model()

# Inputs to the model
__inputs__ = [torch.randn(1, 3, 256, 32),
  torch.randn(1, 3, 96, 2),
  torch.randn(1, 3, 234, 129),
  torch.randn(1, 3, 34, 32),
  torch.randn(1, 3, 232, 32),
  torch.randn(1, 3, 96, 64),
  torch.randn(1, 3, 3, 45),
  torch.randn(1, 3, 232, 2),
  torch.randn(1, 3, 231, 6),
  torch.randn(1, 3, 76, 123),
  torch.randn(1, 3, 36, 234),
  torch.randn(1, 3, 34, 32),
  torch.randn(1, 3, 45, 64),
  torch.randn(1, 3, 237, 4),
  torch.randn(1, 3, 46, 19),
  torch.randn(1, 3, 232, 3),
  torch.randn(1, 3, 3, 2)]

__output_size__ = 1

