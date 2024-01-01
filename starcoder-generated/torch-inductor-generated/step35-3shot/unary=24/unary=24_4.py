
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(32, 8, 3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(8, 8, 1)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), groups=1, bias=False)
    def forward(self, x):
        negative_slope = 0.5
        v1 = self.conv0(x)
        v2 = self.conv1(v1)
        v3 = torch.relu(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose(v4)
        v6 = v5 > 0  # type: torch.Tensor
        v7 = v5 * negative_slope  # type: torch.Tensor
        v8 = torch.where(v6, v5, v7)  # type: torch.Tensor
        return v8
x1 = torch.randn(1, 32, 128, 128)
