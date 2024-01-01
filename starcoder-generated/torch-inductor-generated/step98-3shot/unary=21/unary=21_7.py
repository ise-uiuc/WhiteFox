
class ModelTanh(torch.nn.Module):
    def __init__(self, input_shape=[1, 3, 224, 224]):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_shape[1], 24, 11, padding=1)
        self.max1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(24, 24, 7, stride=2, padding=0)
        self.max2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(24, 24, 5, stride=2, padding=0)
        self.max3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(24, 24, 3, padding=0)
        self.max4 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.conv5 = torch.nn.Conv2d(24, 1, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.max1(v1)
        v3 = self.conv2(v2)
        v4 = self.max2(v3)
        v5 = self.conv3(v4)
        v6 = self.max3(v5)
        v7 = self.conv4(v6)
        v8 = self.max4(v7)
        v9 = self.dropout(v8)
        v10 = self.conv5(v9)
        v10 = self.tanh(v10)
        return v10
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
