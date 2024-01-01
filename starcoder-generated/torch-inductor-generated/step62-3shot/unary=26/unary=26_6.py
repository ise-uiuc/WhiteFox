
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(32, 64, 3, stride=1, padding=1, dilation=1)
        self.conv_t = torch.nn.ConvTranspose1d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.batch_norm = torch.nn.BatchNorm3d(64)
    def forward(self, x2):
        x3 = self.conv1(x2)
        x4 = self.batch_norm(x3)
        x5 = self.conv_t(x4)
        x6 = self.batch_norm(x5)
        h1 = F.leaky_relu(x6, negative_slope=0.1)
        h2 = torch.nn.functional.interpolate(h1, size=64)
        return h2
# Inputs to the model
x2 = torch.randn(1, 32, 64)
