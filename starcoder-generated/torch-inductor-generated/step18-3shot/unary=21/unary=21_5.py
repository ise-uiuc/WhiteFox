
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_19 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x19):
        x20 = self.conv_19(x19)
        x21 = torch.tanh(x20)
        return x21
# Inputs to the model
x19 = torch.randn(1, 1, 169, 86)
