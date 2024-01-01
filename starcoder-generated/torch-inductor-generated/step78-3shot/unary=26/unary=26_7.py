
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_1 = torch.nn.ConvTranspose2d(72, 17, 3, stride=1, padding=0, bias=True)
        self.relu_1 = torch.nn.ReLU()
        self.conv_t_2 = torch.nn.ConvTranspose2d(17, 74, 1, stride=1, padding=0, bias=True)
    def forward(self, x):
        i0 = self.conv_t_1(x)
        i1 = self.relu_1(i0)
        i2 = self.conv_t_2(i1)
        return i2
# Inputs to the model
x = torch.randn(7, 72, 45, 13) # This model is not used in the tutorial
