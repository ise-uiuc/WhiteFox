
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv = torch.nn.Conv2d(3, 64, 7, same_padding=False)
    def forward(self, x1):  
        v1 = self.sigmoid(self.conv(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
