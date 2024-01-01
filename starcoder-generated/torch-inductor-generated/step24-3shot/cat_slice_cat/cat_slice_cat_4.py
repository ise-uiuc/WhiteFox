
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x1 = torch.cat([x1, x1], dim=1)
        v1 = x1[:, 0:9223372036854775807]
        x2 = torch.cat([x1, x2], dim=1)
        v3 = x2[:, 0:size]
        return v1, v3



