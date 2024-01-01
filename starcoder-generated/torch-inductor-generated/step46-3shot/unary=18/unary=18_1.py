
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=512, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.pool1 = torch.nn.AvgPool2d(kernel_size=(1, 3))
        self.conv2 = torch.nn.Conv2d(512, 512, (1, 1), stride=(1, 1), padding=0)
        self.pool2 = torch.nn.AvgPool2d(kernel_size=(1, 3))
        self.linear1 = torch.nn.Linear(in_features=25088, out_features=11)
        # self.linear2 = torch.nn.Linear(in_features=13, out_features=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.pool1(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = self.pool2(v5)
        v7 = v6.view(-1, 25088)
        v8 = self.linear1(v7)
        # v9 = self.linear2(v8)
        return v8
# Inputs to the model
x1 = torch.randn(1, 12, 275, 487)
