
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v2 = torch.cat([x1 for _ in range(10)])
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:64]
        