
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = None
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = torch.from_numpy(np.array([[[0.4567, -0.8451, 0.9851]], [[1.6432, 0.2345, -0.1234]]])).float().requires_grad_(True)
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
