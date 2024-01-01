
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 3)
        torch.manual_seed(1)
        self.layernorm = torch.nn.LayerNorm(3)
    def forward(self, x2):
        x = self.conv(x2)
        y = self.layernorm(x)
        return y
# Inputs to the model
x2 = torch.randn(1, 3, 6, 6)
