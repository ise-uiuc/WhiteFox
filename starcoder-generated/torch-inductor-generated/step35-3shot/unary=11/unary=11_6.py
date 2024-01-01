
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 5, dilation=2, bias=False)
        self.dense = torch.nn.Linear(264, 128)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = v5.flatten(start_dim=1, end_dim=-1)
        v7 = self.dense(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
