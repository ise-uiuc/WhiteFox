
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 203), stride=(1, 1),
                                    padding=(0, 0), dilation=(1, 1))
    def forward(self, input):
        X0 = self.conv_1(input)
        X1 = torch.tanh(X0)
        return X1
# Inputs to the model
input = torch.randn(1, 3, 253, 1)
