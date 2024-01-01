
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        t1 = torch.cat([x1, x2], dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:x2.size(2)]
        t4 = torch.cat([t1, t3], dim=1)
        v1 = t4 + t3 + t2 + t1 + x1 + x2 + x3
        v2 = torch.mean(v1, dim=4)
        v3 = torch.mean(v2, dim=3)
        v4 = torch.mean(v3, dim=2)
        v5 = torch.mean(v4, dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 784, 1, 1)
x2 = torch.randn(1, 784, 40, 27)
x3 = torch.randn(1, 784, 1, 71)
