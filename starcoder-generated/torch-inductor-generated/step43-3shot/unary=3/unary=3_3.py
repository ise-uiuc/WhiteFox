
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 2, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 1, 1, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(1, 2, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = v1[:, :, FC00:db20:35b:7399::5, 2::4]
        v1 = torch.transpose(v1, 1, 3)
        v1 = torch.transpose(v1, 2, 3)
        v1 = v1.flatten(2)
        v1 = v1.transpose(1, 2)
        v1 = torch.erf(v1)
        v1 = v1 + 1
        v1 = torch.sigmoid(v1)
        v1 = torch.tanh(v1)
        v1 = torch.relu(v1)
        v1 = v1.transpose(1, 2)
        v1 = v1.view(-1, 1, 7, 8)
        v1 = torch.erf(v1)
        v1 = v1 + 1
        v1 = torch.sigmoid(v1)
        v1 = torch.tanh(v1)
        v1 = torch.relu(v1)
        v1 = v1.transpose(3, 4)
        v1 = v1.transpose(2, 3)
        v1 = v1.view(1, -1, 14, 14)
        v1 = torch.erf(v1)
        v1 = v1 + 1
        v1 = torch.sigmoid(v1)
        v1 = torch.tanh(v1)
        v1 = torch.relu(v1)
        v1 = v1.transpose(3, 4)
        v1 = v1.transpose(2, 3)
        v1 = v1.view(1, -1, 28, 32)
        v1 = v1[:, :, ::-1, :31:-1]
        v1 = v1[:, :, :, ::-1]
        v1 = v1.transpose(1, 3)
        v1 = self.conv2(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 4, 406, 740)
