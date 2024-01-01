
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 8, kernel_size=7, stride=1, padding=3, dilation=2, groups=1, bias=False)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(8, 8, kernel_size=1, stride=2, padding=0, dilation=1, groups=1, bias=False)
        self.conv3 = torch.nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=3, dilation=1, groups=1, bias=False)
        self.conv4 = torch.nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=4, dilation=4, groups=1, bias=False)
        self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.conv5 = torch.nn.Conv1d(8, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv6 = torch.nn.Conv1d(8, 1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
    def forward(self, x3):
        v1 = self.conv1(x3)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v2)
        v5 = self.conv4(v4)
        v6 = self.prelu(v5)
        v7 = self.conv5(v6)
        v8 = self.conv6(v6)
        v9 = v7 * v8
        return v9
# Inputs to the model
x3 = torch.randn(1, 1, 53)
