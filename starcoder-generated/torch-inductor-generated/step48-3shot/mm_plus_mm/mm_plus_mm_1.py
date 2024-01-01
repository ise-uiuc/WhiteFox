
class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.fc1 = nn.Linear(10, 42)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc1(x)
        y3 = self.fc1(x)
        y3 = torch.mm(y3, y3)
        y4 = torch.mm(y3, y4)
        return (y1 + y2 + y3 + y4)
# Inputs to the model
x = torch.randn(10)
