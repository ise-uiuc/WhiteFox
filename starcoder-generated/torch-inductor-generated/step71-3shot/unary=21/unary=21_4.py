
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation =(1,1))
    def forward(self, x):
        v = self.conv(x)
        v2 = torch.tanh(v)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
