
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 10, 2, 1)
        self.conv2 = torch.nn.Conv2d(10, 10, 10, 2, 1)
        self.conv3 = torch.nn.Conv2d(10, 10, 10, 2, 1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.layer1 = torch.nn.Sequential(self.conv1, self.relu1)
        self.layer2 = torch.nn.Sequential(self.conv2, self.relu2)
        self.layer3 = torch.nn.Sequential(self.conv3, self.relu3)
    def forward(self, input):
        out1 = self.layer1(input)
        out2 = self.layer2(input)
        out3 = self.layer3(input)
        outs = [out1, out2, out3]
        return outs
# Inputs to the model
x1 = torch.randn(10, 1, 112, 112)
