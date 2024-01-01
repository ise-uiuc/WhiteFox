
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 127, 5, stride=5, bias=False)
        self.conv2 = torch.nn.Conv2d(127, 127, 1, bias=True)
        self.conv_t = torch.nn.ConvTranspose2d(127, 127, 5, stride=5)
    def forward(self, x0):
        m1 = torch.nn.functional.leaky_relu(self.conv1(x0))
        f1 = torch.nn.functional.relu(self.conv2(m1))
        f2 = torch.nn.functional.relu(self.conv_t(f1))
        return f2
# Inputs to the model
x0 = torch.randn(3, 1, 5, 4)
