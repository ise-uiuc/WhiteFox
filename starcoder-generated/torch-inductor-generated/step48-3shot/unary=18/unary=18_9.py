
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=69, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=27, out_channels=10, kernel_size=5, stride=3, padding=4)
        self.conv3 = torch.nn.Conv2d(in_channels=10, out_channels=68, kernel_size=7, stride=7, padding=3)
        self.conv4 = torch.nn.Conv2d(in_channels=68, out_channels=13, kernel_size=1, stride=6, padding=5)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        t4 = self.conv4(t3)
        t5 = torch.tanh(t4)
        t6 = torch.sigmoid(t5)
        return t6.flatten()
# Inputs to the model
x1 = torch.randn(1, 69, 100, 100)
