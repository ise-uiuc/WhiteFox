
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv6 = torch.nn.ConvTranspose2d(1, 3, 3, stride=2, padding=1, output_padding=0)
        self.conv7 = torch.nn.Conv2d(3, 1, 3, stride=1)
        self.conv8 = torch.nn.Conv2d(1, 3, 3, stride=1)
        self.conv9 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        x3 = self.conv3(x1)
        x4 = torch.tanh(x3)
        x5 = self.conv6(torch.cat((x2, x4), 1))
        x6 = self.conv7(x5)
        x7 = torch.tanh(x6)
        x8 = self.conv8(x7)
        x9 = torch.tanh(x8)
        x10 = self.conv9(torch.cat((x5, x9), 1))
        x11 = torch.tanh(x10)
        return x11
# Inputs to the model
x = torch.rand(1, 3, 49, 89)
