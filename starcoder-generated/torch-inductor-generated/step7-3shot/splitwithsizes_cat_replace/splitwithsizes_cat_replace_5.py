
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        bs, c, h, w = x1.size()
        assert c % 5 == 0
        split = torch.split(x1, [int(c/5) for _ in range(5)], dim=1)
        combined = []
        for i in range(5):
          combined.append(split[i] * i)
        return torch.cat(combined, dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
