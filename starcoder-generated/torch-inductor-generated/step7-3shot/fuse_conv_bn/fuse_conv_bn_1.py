
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(kernel_size=1, in_channels=16, out_channels=8, groups=3)
        self.bn1 = nn.BatchNorm1d(num_features=8, eps=9., affine=False)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_output = self.conv1(x)
        bn1_output = self.pool1(self.bn1(conv1_output))
        return self.relu1(bn1_output)
# Inputs to the model
x = torch.randn(2, 8, 4, 3)
