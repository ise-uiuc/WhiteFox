
# Description of requirements:
# Add the convolution layer to the model. When adding the convolution layer, make sure stride is a single positive integer, kernel size is a tuple that contains 1 positive integer, and padding is a string representing a padding convention in PyTorch.

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=2, padding='same')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
model = Model()
