
class Model(torch.nn.Module):
    def __init__(self, min_value=1.5, max_value=1.7):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 64, 5, stride=2, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(64,64, 3, stride=2, padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(64,512, 1, stride=1, padding=0)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(512,1280, 4, stride=4, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v4 = self.conv_transpose_4(v3)
        v5 = torch.clamp_min(v4, self.min_value)
        v6 = torch.clamp_max(v5, self.max_value)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 12, 15)
