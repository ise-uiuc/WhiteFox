
class Model(torch.nn.Module):
    def __init__(self, min_value=0.43, max_value=1.76):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(8, 12, (1, 3, 3), stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
        self.act_11 = torch.nn.ReLU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.gelu(x1)
        v2 = self.conv_transpose3d(v1)
        v3 = self.tanh(v2)
        v4 = self.act_11(v3)
        v5 = (self.max_value + 0.031970393009503294) * v4
        v6 = torch.clamp_min(v5, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        return v7
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64, 64)
