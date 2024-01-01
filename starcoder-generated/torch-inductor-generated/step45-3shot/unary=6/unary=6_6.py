
class Model_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(3, 3, kernel_size=3, padding=[2, 1, 1], bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool3d = torch.nn.MaxPool3d(kernel_size=2, padding=0, stride=1, dilation=1, ceil_mode=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout3d(p=0)
    def forward(self, x1):
        v1 = self.conv3d(x1)
        v2 = self.relu(v1)
        v3 = torch.max(v2, dim=2, keepdim=True)
        v4 = torch.mean(v3, dim=2, keepdim=True)
        v5 = self.sigmoid(v4)
        v6 = self.dropout(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, 64)
