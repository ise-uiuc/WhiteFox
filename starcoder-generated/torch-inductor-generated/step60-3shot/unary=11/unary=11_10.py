
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(9, 9, 4, stride=2)
        self.conv = torch.nn.Conv2d(9, 64, 5, stride=1, padding=2)
        self.max_pool2d = torch.nn.MaxPool2d(2)
    def forward(self, x1) -> torch.Tensor:
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        v7 = self.max_pool2d(v6)
        # add one more layer before the end
        v8 = torch.nn.Dropout(p=0.1)(v7)
        return v8
# Inputs to the model
x1 = torch.randn(2, 9, 64, 64)
