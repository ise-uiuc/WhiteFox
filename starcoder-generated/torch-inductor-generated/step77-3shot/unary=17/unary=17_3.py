
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=(1, 3), padding=(0, 1), groups=1, bias=False, stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(3, 6, kernel_size=(1, 3), stride=2, groups=1, bias=False, padding=(0, 1))
        self.conv3 = torch.nn.ConvTranspose2d(6, 9, kernel_size=(5, 5), stride=1, groups=1, bias=False, padding=(2, 2))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 1000, 2000)
