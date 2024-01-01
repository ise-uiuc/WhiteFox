
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = torch.nn.Conv2d(1, 4, 7)
        self.conv2 = torch.nn.Conv2d(4, 2, 5)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.fc = torch.nn.Linear(84, 256)
        self.relu = torch.nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()
        self.conv3 = torch.nn.Conv2d(2, 1, 5)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        # TODO (soto) - add more layers here
        x = self.conv3(x)
        return self.sigmoid(self.dequant(x))
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
