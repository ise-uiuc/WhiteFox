
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad1 = torch.nn.ZeroPad2d((1, 1, 1, 1))
        self.pad2 = torch.nn.ZeroPad2d((1, 1, 1, 0))
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.pad1(x1)
        v2 = self.conv1(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.pad2(v3)
        return v4
# Inputs to the mode
x1 = torch.Tensor(16, 1, 224, 1792)
