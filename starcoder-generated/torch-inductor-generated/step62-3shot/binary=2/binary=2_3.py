
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, np.random.randint(1, 5), 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(np.random.randint(1, 5), 8, 1, stride=1, padding=0) # use self.conv1.out_channels or a randomly generated number
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v4 = v2 - 7000
        return v4
# Inputs to the model
x = torch.randn(1, 3, 8, 8)
