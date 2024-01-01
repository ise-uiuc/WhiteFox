
class ModelTanh(torch.nn.Module):
    def __init__(self, in_channels, out_channels): 
        super(ModelTanh, self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(in_channels, 4, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(4, out_channels, 1, 1, 0)
    def forward(self, x1, x2):
        x3 = self.conv1(x1)
        x4 = self.relu(x3)
        x5 = self.conv2(x4)
        x6 = torch.tanh(x5)
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
x2 = torch.randn(1, 3, 5, 5)
