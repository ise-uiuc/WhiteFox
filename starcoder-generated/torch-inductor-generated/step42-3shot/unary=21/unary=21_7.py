
class ModuleTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(in_channels=1, out_channels=2, stride=(1, 1), kernel_size=2)
        self.relu1 = torch.nn.ReLU6(inplace=False)
    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = self.relu1(conv1)
        return relu1
x = torch.randn(1, 1, 5, 5, 5)
