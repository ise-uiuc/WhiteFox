
class Model(nn.Module):
    def __init__(self, channels):
        super(Model, self).__init__()
        self.conv = nn.Conv1D(channels=channels, kernel_size=2)
    def forward(self, x, y):
        h1 = self.conv(x).sum(dim=2).reshape(x.shape[0], x.shape[1], 1)
        h2 = self.conv(y).reshape(y.shape[0], y.shape[1])
        return h1 + h2
# Inputs to the model
input1 = torch.randn(1, 1, 32)
input2 = torch.randn(1, 2, 24)
