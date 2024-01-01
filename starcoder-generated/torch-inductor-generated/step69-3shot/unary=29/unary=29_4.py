
class Model(torch.nn.Module):
    def __init__(self, min_value=[0.4376, 0.9040], max_value=[0.9751, 0.4770]):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(2, 17, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        min_value = torch.tensor(self.min_value)
        max_value = torch.tensor(self.max_value)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 16)
