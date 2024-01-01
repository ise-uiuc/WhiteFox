
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
 
    def forward(self, x1, x2, x):
        c1 = torch.cat(input_tensors, dim=1)
        s1 = c1[:, 0:9223372036854775807]
        s2 = s1[:, 0:x.size()]
        c2 = torch.cat([c1, s2], dim=1)
        return c2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 32)
x = torch.randn(1, 3, 64)
