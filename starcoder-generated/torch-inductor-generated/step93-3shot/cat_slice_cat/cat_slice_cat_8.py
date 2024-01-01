
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5, x6, size):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4
 
size = torch.randint(1, 33, (1,), dtype=torch.int64)[0]
x1 = torch.randn(1, 64, 9223372036854775807)
x2 = torch.randn(1, 64, 9223372036854775807)
x3 = torch.randn(1, 64, 9223372036854775807)
x4 = torch.randn(1, 64, 9223372036854775807)
x5 = torch.randn(1, 64, 9223372036854775807)
x6 = torch.randn(1, 64, 9223372036854775807)
