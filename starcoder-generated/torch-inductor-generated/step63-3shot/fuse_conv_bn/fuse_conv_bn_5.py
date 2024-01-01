
class Model(torch.nn.Module):
    class Simple1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5), stride=(2), padding=(1), dilation=())
            self.relu = torch.nn.ReLU(inplace=False)

        def forward(self, input):
            identity = input
            out = self.conv1(input)
            out = self.relu(out)
            return out

    def __init__(self):
        super().__init__()
        self.layer1 = self.Simple1()
        self.layer2 = self.Simple1()
        self.layer3 = self.Simple1()

    def forward(self, input):
        x1 = self.layer1(input)
        x2 = self.layer2(x1)
        out = self.layer3(x2)
        return out
# Inputs to the model
input = torch.randn(1, 1, 5, 5)
