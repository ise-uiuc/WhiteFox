
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 300, 4, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(300, 300, 1)
        self.conv3 = torch.nn.Conv2d(300, 300, 1)
        self.conv4 = torch.nn.Conv2d(300, 300, 1)
        self.conv5 = torch.nn.Conv2d(300, 300, 1)
        self.conv6 = torch.nn.Conv2d(300, 300, 1)
        self.conv7 = torch.nn.Conv2d(300, 300, 1)
        self.conv8 = torch.nn.Conv2d(300, 300, 1)
        self.conv9 = torch.nn.Conv2d(300, 300, 1)
        return
    def forward(self, z):
        r1 = torch.relu(self.conv1(z))
        r3 = torch.relu(self.conv3(r1))
        r5 = torch.relu(self.conv5(r3))
        r7 = torch.relu(self.conv7(r5))
        r9 = torch.relu(self.conv9(r7))
        r300 = r9
        x = torch.tanh(r300)
        return
# Inputs to the model
z = torch.randn(1, 3, 64, 64)
