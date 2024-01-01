
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 50, (3, 2), stride=(3, 2), padding=(2, 1))
    def forward(self, x1):
        v7 = torch.conv1d(x1, weight=torch.randn(50, 1, 2), padding='same')
        v8 = v7 + 3
        v9 = torch.clamp_min(v8, 0)
        v10 = torch.clamp_max(v9, 6)
        v11 = v10 / 6
        return v11
# Inputs to the model
x1 = torch.randn(1, 1, 64)
