
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 7, stride=1, padding=3)
        self.conv1 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
    def forward(self, inputs):
        v = self.conv0(inputs)
        v = self.conv1(v)
        v = self.conv2(v)
        v = self.conv3(v)
        v = self.conv4(v)
        return torch.mean(v, dim=[2, 3])
# Inputs to the model
input_data = torch.randn(2, 3, 224, 224)
