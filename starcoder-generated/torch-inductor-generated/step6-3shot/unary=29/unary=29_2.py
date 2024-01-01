
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.28, max_value=0.28):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 2, bias=True, stride=2, padding=1)
        self.linear1d_1 = torch.nn.Linear(7168, 192)
        self.linear1d_2 = torch.nn.Linear(192, 64)
        self.linear1d_3 = torch.nn.Linear(64, 8)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v0 = x1.size()
        v1 = self.conv_transpose(x1)
        v2 = v1.reshape(v0[0] * v0[1], 7168)
        v3 = self.linear1d_1(v2)
        v4 = torch.relu(v3)
        v5 = self.linear1d_2(v4)
        v6 = torch.relu(v5)
        v7 = self.linear1d_3(v6)
        v8 = torch.clamp_min(v7, self.min_value)
        v9 = torch.clamp_max(v8, self.max_value)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
