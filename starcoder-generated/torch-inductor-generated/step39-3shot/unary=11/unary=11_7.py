
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if config.get('weight'):
            self.conv_transpose = torch.nn.ConvTranspose2d(config.get('in_features'), config.get('out_features'), 3, stride=1, padding=1)
        else:
            self.conv_transpose = torch.nn.ConvTranspose2d(1024, 2048, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 1024, 25, 25)
