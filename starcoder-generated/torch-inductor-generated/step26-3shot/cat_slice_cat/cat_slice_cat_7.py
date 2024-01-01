
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x1):
        x2 = torch.cat([x1, x1], dim=1)
        x3 = x2[:, 0:9223372036854775807]
        x4 = x3[:, 0:self.size]
        x5 = torch.cat([x1, x4], dim=1)
        return x5

