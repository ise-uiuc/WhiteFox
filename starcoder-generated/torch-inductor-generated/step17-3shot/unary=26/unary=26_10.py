
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(480, 24, kernel_size=(3,3), padding=1, stride=2)
        self.linear1 = torch.nn.Linear(256, 345)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(324, 2)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = self.linear1(torch.flatten(x2))
        x4 = self.relu(x3)
        x5 = self.linear2(x4)
        return x5
# Inputs to the model
x1 = torch.randn(16, 480, 6, 6)
