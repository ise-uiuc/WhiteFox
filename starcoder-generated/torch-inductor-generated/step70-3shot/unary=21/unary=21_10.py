
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.Conv2d(1,1,1)
    def forward(self,x):
        x = self.conv_t(x)
        # The next line is expected to be detected as pattern.
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1,2,256,256)
