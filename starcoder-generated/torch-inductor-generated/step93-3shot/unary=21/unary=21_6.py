
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        s2 = v1.shape[1]
        v2 = torch.tanh(v1[:, 0:1, :, :]) + torch.tanh(v1[:, 1:2, :, :]) + \
             torch.tanh(v1[:, 2:3, :, :]) + torch.tanh(v1[:, 3:4, :, :])
        return v2
# Inputs to the model
x = torch.randn(1024, 1, 32, 32)
