
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 1024)
        self.bn = torch.nn.BatchNorm1d(1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.avg_pool1 = torch.nn.AdaptiveAvgPool1d(1)# torch.nn.AvgPool1d(5, stride=4, padding=0, ceil_mode=False, count_include_pad=True)
        self.avg_pool2 = torch.nn.AdaptiveAvgPool1d(1)
        self.fc3 = torch.nn.Linear(1024, 4)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(1, 4, requires_grad=True)
