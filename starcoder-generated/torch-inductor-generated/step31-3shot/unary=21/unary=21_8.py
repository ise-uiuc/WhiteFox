
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 256, 1, padding=(0,1), stride=1)
        self.conv2 = torch.nn.Conv2d(256, 256, (1, 5), padding=(0,2), stride=1)
        self.conv3 = torch.nn.Conv2d(256, 3, (1, 7), padding=(0,3), stride=1)
    def forward(self, x1):
        v2 = self.conv1(x1)
        v2 = torch.tanh(v2)
        v2 = self.conv2(v2)
        v2 = torch.tanh(v2)
        v2 = self.conv3(v2)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(4, 3, 256, 241)
