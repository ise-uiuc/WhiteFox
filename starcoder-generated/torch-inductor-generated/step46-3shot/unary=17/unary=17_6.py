
class Model(torch.nn.Module):
    def __init__(self):
        stride = 2
        stride2 = 16
        stride3 = stride2*stride
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 3, stride=stride)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(int((32 * stride2 * stride2)/16 * stride3), 2)
        self.maxpool = torch.nn.MaxPool2d(3, stride=stride)
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = self.relu1(v1)
        v3 = v2.view(-1)
        v4 = self.fc(v3)
        v5 = v4.view(1,-1)
        v6 = self.maxpool(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 3, 24, 24)
