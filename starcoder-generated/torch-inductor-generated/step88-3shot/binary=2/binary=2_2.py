
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 20, 5, stride = 1,padding = 0).requires_grad_(False)
        self.avgpool = torch.nn.AvgPool2d(2)
        self.conv1x1_1 = torch.nn.Conv2d(20, 128, (1, 1), stride = 1,padding = 0)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, input_1):
        v1 = self.conv2d(input_1)
        v2 = self.avgpool(v1)
        v3 = self.conv1x1_1(v2)
        v4 = self.relu(v3)
        t1 = v4.view(1, -1)
        return t1
# Inputs to the model
input_1 = torch.randn(1, 1, 224, 224)
