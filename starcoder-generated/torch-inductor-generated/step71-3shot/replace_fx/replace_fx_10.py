
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 3)
    def forward(self, x):
        x = self.conv(x)
        t1 = torch.pow(x, 3)
        dropout1 = F.dropout(t1, p=0.05)
        t2 = torch.rand_like(t1)
        return dropout1
# Inputs to the model
x = torch.randn((1, 3, 3, 3))
