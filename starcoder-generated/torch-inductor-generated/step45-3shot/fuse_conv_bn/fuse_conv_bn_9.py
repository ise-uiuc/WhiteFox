
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.CBR(2, 2, 1)
    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(1, 4, 4, 4)
