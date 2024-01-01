
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, size):
        l1 = [x1, x2]
        t1 = torch.cat(l1, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:size]
        l2 = [x3, x4]
        t4 = torch.cat(l2, dim=1)
        t5 = torch.cat([t1, t3], dim=1)
        return t5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 4, 64, 64)
x3 = torch.randn(1, 8, 64, 64)
x4 = torch.randn(1, 4, 64, 64)
size = 5
