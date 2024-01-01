
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)
        self.softmax = torch.nn.Softmax(dim = 2)

    def forward(self, input):
        h = self.conv1(input)
        h = self.softmax(h)
        return h
# Inputs to the model
x1 = torch.randn(1, 1, 64)
