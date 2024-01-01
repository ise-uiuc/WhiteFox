
class Model(torch.nn.Module):
    def __init__(self, min_value=0.9, max_value=2.6):
        super().__init__()
        self.sigmoid = torch.nn.Softplus()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 16, 1, stride=1, padding=1)
        self.act_4 = torch.nn.hardtanh()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x5):
        v6 = self.conv_transpose(x5)
        v7 = torch.clamp_min(v6, self.min_value)
        v8 = torch.clamp_max(v7, self.max_value)
        v9 = self.sigmoid(v8)
        return v9
# Inputs to the model
x5 = torch.randn(1, 16, 124, 124)
