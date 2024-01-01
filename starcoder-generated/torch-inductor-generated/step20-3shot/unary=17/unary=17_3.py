
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.ConvTranspose2d(3, 32, 3, stride=2, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose3d(32, 32, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.t1(x1) # apply pointwise convolution to input tensor x1
        v2 = torch.relu(v1)
        v3 = self.conv_transpose(v2) # now apply pointwise transposed convolution to v2
        v4 = torch.tanh(v3) # finally apply activation function to output of transposed convolution
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
x2 = torch.randn(1, 3, 128, 128)
