
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=1)
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x).permute(0, 2, 3, 1).permute(0, 3, 1, 2)
        return self.relu2(self.conv3(x))
# Inputs to the model
x = torch.randn(2, 3, 300, 300)
