
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(255, 1, (3, 1), stride=1, padding=(1, 1), output_padding=(1, 0), bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(256, 1, (4, 1), stride=1, padding=(2, 1), output_padding=(2, 0), bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(129, 1, (6, 1), stride=1, padding=(3, 1), output_padding=(3, 0), bias=False)
        self.conv3 = torch.nn.ConvTranspose2d(7, 9, (7, 1), stride=1, padding=(4, 1), output_padding=(4, 0), bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(x1)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        v9 = torch.sigmoid(v7)
        return v9
# Inputs to the model
x1 = torch.randn(1, 256, 1, 1)
