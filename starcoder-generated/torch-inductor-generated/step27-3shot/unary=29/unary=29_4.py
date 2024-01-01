
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=0.6):
        super().__init__()
        self.conv_transpose2d_2 = torch.nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv_transpose2d_3 = torch.nn.ConvTranspose2d(16, 32, 5, stride=2, padding=2)
        self.conv_transpose2d_4 = torch.nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v5 = self.conv_transpose2d_2(x3)
        v8 = self.conv_transpose2d_3(v5)
        v11 = self.conv_transpose2d_4(v8)
        v13 = v11 + 4.7809789086016156
        v15 = torch.clamp_min(v13, self.min_value)
        v17 = torch.clamp_max(v15, self.max_value)
        return v17
# Inputs to the model
x3 = torch.randn(1, 16, 224, 224)
