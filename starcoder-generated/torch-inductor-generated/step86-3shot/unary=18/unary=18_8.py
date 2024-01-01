
class TestNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2, stride=2)
        self.conv1 = torch.nn.Conv2d(1, 2, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 4, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 2, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(2, 1, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(1, 2, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv3d(1, 1, kernel_size=3, padding=2)
    def forward(self, x):
        x = self.maxpool(torch.nn.Sigmoid()(torch.nn.MaxPool3d(kernel_size=2, stride=1, padding=1)(self.conv6(x))))
        x = torch.nn.ReLU()(self.conv1(x))
        x = torch.nn.MaxPool2d(2, stride=2)(torch.nn.Sigmoid()(self.conv2(x)))
        x = torch.nn.Softplus()(self.conv3(x))
        x = torch.nn.AdaptiveMaxPool2d(3)(torch.nn.Softmax2d()(torch.nn.Conv3d(2, 1, kernel_size=2, padding=(1, 1))(x)))
        x = torch.nn.LogSigmoid()(self.conv4(x))
        x = torch.nn.AdaptiveMaxPool1d(3)(torch.nn.Softmax(dim=1)(torch.nn.Conv1d(1, 2, kernel_size=2, padding=1)(x)))
        x = torch.nn.Tanh()(self.conv5(x))
        return x

model = TestNetwork()
# Input to the model
x = torch.randn(4, 1, 64, 64, 64)
