
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(). __init__()
        self.upsample2D = F.interpolate
        self.conv1 = torch.nn.Conv2d(32, 8, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=1)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=2)
    def forward(self, input_tensor):
        x = torch.tanh(self.conv1(input_tensor))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = self.upsample2D(x, size=[4, 4], mode='bilinear', align_corners=False)
        x = self.conv4(x)
        x = torch.tanh(self.conv5(x))
        return x
# Inputs to the model
x = torch.randn(1, 32, 6, 6)
