
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(92, 3, kernel_size=(5, 45), bias=True, stride=(36047394, 832226597), padding=(0, 5))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(36047394, 832226597, 92, 2)
