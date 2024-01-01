
class Model(torch.nn.Module):
    def __init__(self, max_value=0.9):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(4, 1, 5, stride=2, padding=2)
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_max(v1, self.max_value)
        return v2
# Inputs to the model
x1 = torch.randn(2, 4, 10)
