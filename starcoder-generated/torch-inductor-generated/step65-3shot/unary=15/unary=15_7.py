
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(256, 256, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.conv2 = torch.nn.Conv3d(256, 256, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv3d(256, 256, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
o1 = torch.randn(17, 256, 10, 10, 10)
