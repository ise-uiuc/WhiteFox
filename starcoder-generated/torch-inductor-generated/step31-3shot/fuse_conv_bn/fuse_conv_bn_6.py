
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv = torch.nn.Conv1d
        bn = torch.nn.BatchNorm1d
        self.in_channels = 3
        self.conv = conv(self.in_channels, 16, kernel_size=7, bias=False)
        self.bn = bn(16, momentum=0.5)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x2):
        x = self.conv(x2)
        x = self.bn(x)
        x = self.relu(x)
        return x
# Inputs to the model
x2 = torch.randn(1, 3, 16)
