
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = torch.nn.Conv2d(1, 11, 11, stride=(2,2), padding=(2,2), dilation=(2,2))
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(11, 16, 5, stride=(1,1), padding=(1,1), dilation=(1,1))
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        x = self.con1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x
# Inputs to the model
input = torch.randn(1, 1, 28, 28)
