
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 2)
        self.batchNorm1 = torch.nn.BatchNorm2d(2)
        self.conv2 = torch.nn.Conv2d(2, 2, 2)
        # This BN is not used in the model. It is only here to verify the pattern match works with multiple BN layers.
        self.batchNorm2 = torch.nn.BatchNorm2d(2)

    def forward(self, x):
        _ = self.conv1(x)
        _ = self.batchNorm1(x)
        _ = self.conv2(x)
        _ = self.batchNorm2(x)
        return x

net = Net()
