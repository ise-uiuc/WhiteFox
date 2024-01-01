
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 64, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 64, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        x2 = x1[:,0:5,:,:] + x1[:,4:6,:,:]
        v2 = self.conv2(x2)
        x3 = torch.stack([x1[:, 0, :, :], x1[:, 3, :, :], x1[:, 1, :, :], x1[:, 4, :, :], x1[:, 2, :, :], x1[:, 5, :, :]], dim=1)
        v3 = self.conv3(x3)
        v4 = v1 + v2
        v5 = torch.clamp_min(v4, 1)
        v6 = torch.clamp_max(v5, 10)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
