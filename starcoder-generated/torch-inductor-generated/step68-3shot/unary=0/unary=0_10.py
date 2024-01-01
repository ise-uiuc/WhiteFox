
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 10, 11, stride=4, padding=0)

    def forward(self, x11, x41):
        x11 = self.conv1(x11)
        x41 = self.conv2(x41)

        x11 = x11 * 0.5
        x41 = x41 * 0.5

        x11 = x11 * x11
        x41 = x41 * x41

        x11 = x11 * x11
        x41 = x41 * x41

        x11 = x11 * 0.044715
        x41 = x41 * 0.044715

        x11 = x11 + x41

        x11 = x11 * 0.7978845608028654
        x41 = x41 * 0.7978845608028654

        x11 = torch.tanh(x11)
        x41 = torch.tanh(x41)

        x11 = x11 + 1
        x41 = x41 + 1

        x11 = x11 * x41

        return x11
# Inputs to the model
x11 = torch.randn(1, 1, 32, 32)
x41 = torch.randn(1, 10, 16, 16)
