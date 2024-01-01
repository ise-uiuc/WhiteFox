
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 26, 2, stride=1, padding=0)
        self.conv1_2 = torch.nn.Conv2d(26, 12, 2, stride=1, padding=1)
        # self.conv1_3 = torch.nn.Conv2d(14, 26, 2, stride=1, padding=0)
        self.relu1_1 = torch.nn.ReLU()
        self.relu1_2 = torch.nn.ReLU()
        self.relu1_3 = torch.nn.ReLU()
        self.relu = torch.nn.ReLU()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(12, 10)
    def forward(self, x):
        x_in = x.clone()
        x_1 = self.conv1_1(x_in)
        # x_1_2 = self.conv1_2(x_1)
        x_1_3 = self.conv1_3(x_1)
        x_1_4 = self.relu1_1(x_1)
        x_1_5 = self.relu1_2(x_1_4)
        x_1_6 = self.avg_pool(x_1_5)
        # x_1_7 = x_1_6.squeeze()
        x_1_7 = x_1_5.flatten()
        x_1_8 = self.fc(x_1_7)
        return x_1_8
        # return x_1_2
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
