
class Model(torch.nn.Module):
    def __init__(self, min_value=-4.0, max_value=0.0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 8, stride=1, padding=3)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.tensor([[[[-0.35711868732452393, 0.5532929682731628, -0.07500041806983948, 1.0675875902175903, -1.3714165687561035, 0.784887261390686, -0.8172100257873535, -0.18377346487045288]]]])
