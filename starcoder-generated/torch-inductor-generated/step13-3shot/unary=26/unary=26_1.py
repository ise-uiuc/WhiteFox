
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(1, 2, (2,), stride=2, padding=(1), bias=False)
        self.relu = torch.nn.ReLU()   
    def forward(self, x1):
        x2 = self.conv_t(x)
        x3 = self.relu(x2)
        x4 = x2 > 0
        x5 = x2 * x3
        x6 = torch.where(x4, x2, x5)
        return x6
# Inputs to the model
x = torch.randn(1, 1, 5, 1)
