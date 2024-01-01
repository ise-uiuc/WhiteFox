
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 8, 3)
        self.conv_2 = torch.nn.Conv2d(8, 8, 3)
        self.conv_3 = torch.nn.Conv2d(8, 8, 3)
        self.conv_4 = torch.nn.Conv2d(8, 1, 3, dtype=torch.float)

        
    def forward(self, x1):
        x2 = self.conv_4(torch.tanh(self.conv_3(torch.tanh(self.conv_2(torch.tanh(self.conv_1(x1)))))))
        return x2
# Input to the model
x1 = torch.randn(1, 1, 28, 28)
