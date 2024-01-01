
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool3d(3, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = v1.permute(0, 2, 3, 1)
        v3 = self.conv(v2).squeeze(3).squeeze(2)
        v4 = torch.reshape(v3, (1, 5, 2))
        return torch.mean(torch.tanh(v4), dim=-1)
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2, 3)
