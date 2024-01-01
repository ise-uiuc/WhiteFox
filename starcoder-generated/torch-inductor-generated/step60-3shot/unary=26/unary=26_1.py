
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose1d(165, 163, 9)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = -self.relu(x2) + self.sigmoid(x2)
        return x3
# Inputs to the model
x1 = torch.randn(4, 165, 46)
