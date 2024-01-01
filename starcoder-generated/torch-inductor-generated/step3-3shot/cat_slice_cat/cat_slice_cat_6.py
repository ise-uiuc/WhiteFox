
class Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x1, x2, x3):
        v1 = torch.cat([x1, x2, x3], dim=1)
        v2 = v1[:, 0:pow(2,63) - 1]
        v3 = v2[:, 0:self.size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4, v3, v2

# Initializing the model
m = Model(1)

# Inputs to the model
x1 = torch.randn(1, pow(2,21), 18, 18)
x2 = torch.randn(1, pow(2,21), 18, 18)
x3 = torch.randn(1, pow(2,21), 18, 18)

__output1__, __output2__, 