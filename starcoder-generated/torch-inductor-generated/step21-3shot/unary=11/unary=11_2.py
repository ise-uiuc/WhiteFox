
# https://discuss.pytorch.org/t/why-does-transpose-conv2d-of-tensor-3-28-32-give-tensor-1-18-28-3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 4, stride=2, padding=1, output_padding=1)
    def forward(self, input):
        return self.conv_transpose(input)
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
