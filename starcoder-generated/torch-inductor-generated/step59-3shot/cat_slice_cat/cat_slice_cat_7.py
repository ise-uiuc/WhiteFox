
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v0 = torch.cat([x1, x2], dim=1)
        v1 = v0[:, 0:x2.shape[1]]
        v2 = v1[:, 0:size]
        v3 = torch.cat([v0, v2], dim=1)
        return v3

m = Model()
x1 = torch.randn(20, 9223372036854775807)
x2 = torch.randn(20, size)
__output___ = m(x1, x2)

