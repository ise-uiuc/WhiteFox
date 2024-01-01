
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.squeeze(torch.squeeze(torch.relu(v1).transpose(-2, -1), -1), -1)
        v3 = self.conv2(v2)
        v4 = torch.squeeze(torch.squeeze(torch.relu(v3).transpose(-2, -1).transpose(-2, -1), -1), -1)
        v5 = self.conv3(v4)
        v6 = torch.squeeze(torch.squeeze(torch.relu(v5).transpose(-2, -1).transpose(-2, -1).transpose(-2, -1), -1), -1)
        v7 = self.conv4(v6)
        v8 = torch.squeeze(torch.squeeze(torch.relu(v7).transpose(-2, -1).transpose(-2, -1).transpose(-2, -1), -1), -1)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
