
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=22, kernel_size=(22, 8), stride=(2, 1), padding=0)
        self.conv1d = torch.nn.Conv1d(in_channels=22, out_channels=22, kernel_size=11, stride=1, padding=5)
        self.convTranspose2d = torch.nn.ConvTranspose2d(in_channels=22, out_channels=3, kernel_size=(3, 9), stride=(2, 2), padding=1)
        self.convTranspose1d = torch.nn.ConvTranspose1d(in_channels=22, out_channels=45, kernel_size=15, stride=1, padding=4)
    def forward(self, x):
        negative_slope = 0.66355
        v1 = self.conv2d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv1d(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        v9 = self.convTranspose2d(v8)
        v10 = v9 > 0
        v11 = v9 * negative_slope
        v12 = torch.where(v10, v9, v11)
        v13 = self.convTranspose1d(v12)
        v14 = v13 > 0
        v15 = v13 * negative_slope
        v16 = torch.where(v14, v13, v15)
        return v16
# Inputs to the model
x = torch.randn(1, 1, 17, 20)
