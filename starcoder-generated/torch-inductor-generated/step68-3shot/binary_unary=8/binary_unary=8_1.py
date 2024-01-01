
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1, bias=False)
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0), dilation=(1, 1), groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1)
        v4 = torch.relu(v2)
        v5 = torch.relu(torch.relu(v1) + torch.relu(v2))
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
