
class Model(torch.nn.Module):
    def __init__(self, min_value=-4, max_value=4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3, stride=1, padding=1).to('cuda:0')
        self.conv2 = torch.nn.Conv2d(6, 8, 3, stride=1, padding=1).to('cuda:0')
        self.relu_1 = torch.nn.ReLU().to('cuda:0')
        self.relu_2 = torch.nn.ReLU6().to('cuda:0')
        self.maxpool_1 = torch.nn.MaxPool2d(3, stride=2, padding=0).to('cuda:0')
        self.conv3 = torch.nn.Conv2d(8, 4, 5, stride=1, padding=0).to('cuda:0')
        self.conv4 = torch.nn.Conv2d(4, 4, 3, stride=2, padding=0).to('cuda:0')
        self.conv5 = torch.nn.ConvTranspose2d(4, 6, 3, stride=2, padding=1).to('cuda:0')
        self.conv6 = torch.nn.ConvTranspose2d(6, 8, 3, stride=1, padding=1).to('cuda:0')
        self.conv7 = torch.nn.ConvTranspose2d(8, 1, 1, stride=1, padding=1).to('cuda:0')
        self.relu_3 = torch.nn.ReLU().to('cuda:0')
        self.relu_4 = torch.nn.ReLU6().to('cuda:0')
        self.maxpool_2 = torch.nn.MaxPool2d(3, stride=2, padding=1).to('cuda:1')
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v2_1 = torch.clamp(v2, self.min_value, self.max_value)
        v3 = self.relu_1(v1)
        v3_1 = self.relu_2(v3)
        v4 = self.maxpool_1(v2_1)
        v5 = self.conv3(v4)
        v6 = self.conv4(v5)
        v7 = self.conv5(v6)
        v8 = self.relu_3(v7)
        v9 = self.conv6(v8)
        v10 = self.relu_4(v9)
        v11 = self.conv7(v10)
        v12 = self.maxpool_2(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 1, 72, 72).to('cuda:0')
