
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, groups=2)
        torch.manual_seed(1)
        self.avg_pool2d = torch.nn.AvgPool2d(3, stride=(2, 2), padding=(1, 1), ceil_mode=False)
        torch.manual_seed(1)
        self.dropout = torch.nn.Dropout(p=0.1)
        torch.manual_seed(3)
        self.linear1 = torch.nn.Linear(2688, 120)
        torch.manual_seed(2)
        self.linear2 = torch.nn.Linear(120, 84)
    def forward(self, x1):
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        x3 = self.avg_pool2d(x2)
        x4 = x3.view(-1, self.num_flat_features(x3))
        x5 = self.linear1(x4)
        x6 = self.dropout(x5)
        x7 = self.linear2(x6)
        x8 = self.relu(x7)
        return x8
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
# Inputs to the model
x1 = torch.randn(1, 3, 50, 50)
