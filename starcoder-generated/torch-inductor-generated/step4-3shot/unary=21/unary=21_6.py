
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (1, 1))
    def forward(self, x2625):
        x2626 = self.conv(x2625)
        x2627 = torch.tanh(x2626)
        return x2627
# Inputs to the model
x2625 = torch.randn(4,1,100,100)
