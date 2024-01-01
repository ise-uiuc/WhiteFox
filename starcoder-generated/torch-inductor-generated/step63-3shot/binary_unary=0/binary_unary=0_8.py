
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        t1 = torch.conv2d(x, y, stride=1, padding=3)
        t2 = torch.conv2d(t1, t1, stride=1, padding=3)
        t3 = torch.conv2d(t2, t2, stride=1, padding=3)
        t4 = t3 + torch.sigmoid(t3)
        return torch.sigmoid(t4)
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
y = torch.randn(16, 16, 7, 7)
