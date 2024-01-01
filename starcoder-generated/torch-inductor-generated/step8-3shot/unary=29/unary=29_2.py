
class Model(torch.nn.Module):
    def __init__(self, min_value=-8.0, max_value=3.6462):
        super().__init__()
        self.Conv2d = torch.nn.Conv2d(5, 9, (6, 1), (2, 1), (0, 0))
        self.gelu = torch.nn.GELU()
        self.Conv2d_1 = torch.nn.ConvTranspose2d(9, 5, (7, 2), (3, 2), (0, 1))
        self.Conv2d_2 = torch.nn.ConvTranspose2d(5, 3, (3, 2), (3, 2), (1, 0))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.Conv2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.gelu(v3)
        v5 = self.Conv2d_1(v4)
        v6 = self.Conv2d_2(v5)
        return torch.clamp_max(v6, 0)
# Inputs to the model
x1 = torch.randn(1, 5, 3, 3)
