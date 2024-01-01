
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 15), stride=(2, 1), padding=(3, 8))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(3, 11), stride=(2, 7), padding=(3, 6))
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(1, 11), stride=(2, 7), padding=(3, 6))
        self.conv4 = torch.nn.Conv2d(128, 32, kernel_size=(3, 11), stride=(2, 7), padding=(3, 6))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = torch.sigmoid(v1)
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv3(v5)
        v7 = torch.sigmoid(v6)
        v8 = self.conv4(v7)
        v9 = torch.sigmoid(v8)
        return v9 
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
