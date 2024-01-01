
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv = torch.nn.Conv2d(8, 1, 1, dtype=torch.float)
    def forward(self, x1):
        x2 = self.conv(torch.tanh(self.conv(torch.tanh(self.conv(torch.tanh(self.conv(torch.tanh(self.conv(torch.tanh(self.conv(x1)))))))))))
        return x2 
# Inputs to the model
x1 = torch.randn(1, 8, 28, 28)
