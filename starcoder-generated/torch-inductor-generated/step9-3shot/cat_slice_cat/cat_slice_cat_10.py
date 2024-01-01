
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2, x3, x4, x5):
        c1 = torch.cat((x1, x2))
        s1 = c1[:, 0:9223372036854775807]
        s2 = s1[:, 0:x1.shape[2]]
        c2 = torch.cat((c1, s2))
        return c2

# Initializing the model
m = Model()

## Inputs to the model
x1 = torch.randn(1, 3, 88, 88)
x2 = torch.randn(1, 3, 88, 88)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 32, 32)
x5 = torch.randn(1, 3, 16, 16)
