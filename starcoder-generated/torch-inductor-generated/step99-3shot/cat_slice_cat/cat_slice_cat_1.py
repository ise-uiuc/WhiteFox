
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t1, t2, t3):
        v1 = torch.cat([t1, t2, t3], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4
 
# Initializing the model
m = Model()
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
x3 = torch.randn(1, 4, 64, 64)
