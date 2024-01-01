
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.stack((x1, x2, x3, x4, x5))
        v2 = v1.permute(1, 0, 2, 3)
        v3 = torch.cat(v2, dim=1)[:, 0:9223372036854775807]
        v4 = v3[:, 0:size]
        v5 = torch.cat([v3, v4], dim=1)
        return v5

