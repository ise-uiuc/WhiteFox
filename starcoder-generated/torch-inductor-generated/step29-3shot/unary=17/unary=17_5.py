
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 32, 3, stride=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(8, 4, 3, stride=1, padding=1)
        self.conv5 = torch.nn.ConvTranspose2d(4, 2, 3, stride=1, padding=1)
        self.convfinal = torch.nn.ConvTranspose2d(2, 4, 3, stride=1, padding=2)
    def forward(self, x, x_len):
        x_len = x_len.reshape(1, -1, 1).float()
        x = torch.mul(x, x_len)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        xn = torch.sigmoid(self.convfinal(x5))
        xn = torch.mul(xn, x_len)
        return xn
# Input for model
x = torch.randn(2, 2, 10, 10)
x_len = torch.randn(2,)
