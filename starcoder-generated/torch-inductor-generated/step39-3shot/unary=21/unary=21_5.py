
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(31, 73, 1, padding=0, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(73, 109, 1, padding=0, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.ConvTranspose2d(109, 45, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.conv4 = torch.nn.Conv2d(45, 15, 1, padding=0, bias=False)
    def forward(self, x17):
        v18 = self.conv1(x17)
        v33 = self.relu1(v18)
        v34 = self.conv2(v33)
        v36 = self.relu2(v34)
        v37 = self.conv3(v36)
        v38 = self.conv4(v37)
        v39 = torch.tanh(v38)
        return v39
# Inputs to the model
x17 = torch.randn(1, 31, 22, 22)
