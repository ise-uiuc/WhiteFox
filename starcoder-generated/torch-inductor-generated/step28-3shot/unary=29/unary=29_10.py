
class Model(torch.nn.Module):
    def __init__(self, min_value=16, max_value=1007):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 34, stride=1, padding=15)
        self.softsign = torch.nn.Softsign()
        self.max_value = max_value
        self.min_value = min_value
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.softsign(v3)
        return v4
# Inputs to the model
x = torch.randn(3, 3, 229, 185)
