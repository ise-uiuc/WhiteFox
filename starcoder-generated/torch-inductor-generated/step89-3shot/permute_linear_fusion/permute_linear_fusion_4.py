
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(num_features=1, eps=1e-05)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.bn2 = torch.nn.BatchNorm2d(num_features=1, eps=1e-05)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0,), dilation=(1,))
        self.flatten = torch.nn.Flatten()
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        x2 = x1.permute(0, 2, 1, 3)
        v1 = self.bn1(x2)
        v2 = v1.permute(0, 2, 1, 3)
        v3 = self.conv1(v2)
        v4 = self.bn2(v3)
        v5 = self.conv2(v4)
        v6 = v5.squeeze(1)
        v7 = self.flatten(v6)
        v8 = self.softmax(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
