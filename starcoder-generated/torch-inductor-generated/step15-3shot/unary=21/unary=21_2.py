
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 23, (3,5), padding=(1,2), dilation=(2,1))
    def forward(self, x17):
        v18 = self.conv(x17)
        v19 = torch.tanh(v18)
        return v19
# Inputs to the model
x17 = torch.randn(3, 20, 16, 16)
