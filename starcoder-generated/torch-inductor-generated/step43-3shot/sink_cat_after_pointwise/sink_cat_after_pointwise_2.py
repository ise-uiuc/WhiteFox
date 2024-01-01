
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, 3)
    def forward(self, x):
        x = self.conv(x)
        x = torch.cat([x, x, x], dim=1)
        x = torch.dropout(x)
        x = torch.relu(-3.0*x)
        x = torch.sigmoid(1.0 + x) if x.shape[1] == 1 else torch.sigmoid(1.0 - x)
        x = torch.sin(x)
        return x
# Inputs to the model
x = torch.randn(1, 10, 3, 11)
