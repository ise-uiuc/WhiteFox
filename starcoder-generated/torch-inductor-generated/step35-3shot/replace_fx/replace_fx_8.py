
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 64, 1)
    def forward(self, x1):
        x1 = F.softmax(x1, dim=-1)
        x2 = F.dropout(x1)
        x3 = self.pool(x1)
        x4 = torch.relu(x2)
        x5 = self.conv(x4)
        x6 = self.convt(x5)
        x7 = torch.relu(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
