
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        kernel_size = 11
        self.conv1 = nn.Conv2d(3, 12, 11, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(3, 12, 11, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(3, 12, 11, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(3, 12, 11, stride=1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(3, 12, 11, stride=1, padding=0, bias=False)
        self.conv6 = nn.Conv2d(3, 12, 11, stride=1, padding=0, bias=False)
        self.conv7 = nn.Conv2d(3, 12, 11, stride=1, padding=0, bias=False)
        self.conv8 = nn.Conv2d(3, 12, 11, stride=1, padding=0, bias=False)
    def forward(self, input1, input2, input3, input4, input5, input6, input7, input8):
        x1 = self.conv1(input1)
        x2 = self.conv2(input2)
        x3 = self.conv3(input3)
        x4 = self.conv4(input4)
        x5 = self.conv5(input5)
        x6 = self.conv6(input6)
        x7 = self.conv7(input7)
        x8 = self.conv8(input8)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        return x
# Inputs to the model
input1 = torch.randn(1, 3, 64, 64)
input2 = torch.randn(1, 3, 64, 64)
input3 = torch.randn(1, 3, 64, 64)
input4 = torch.randn(1, 3, 64, 64)
input5 = torch.randn(1, 3, 64, 64)
input6 = torch.randn(1, 3, 64, 64)
input7 = torch.randn(1, 3, 64, 64)
input8 = torch.randn(1, 3, 64, 64)
