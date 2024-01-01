
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, (3, 4))
    def forward(self, x):
        y2 = self.conv(x)
        y2 = y2.view(y2.shape[3], y2.shape[2], y2.shape[1], y2.shape[0])
        y2 = torch.cat([y2, y2, y2], 1)
        y2 = y2.permute(0, 3, 1, 2)
        return y2
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
# Model Ends