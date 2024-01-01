
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=False)
 
    def forward(self, x1, x2):
        y1, y2 = self.fc(x1).chunk(2, dim=1)
        y3 = (x2 + y2) - y1
        return y3

x1 = torch.randn(1, 16)
x2 = torch.randn(1, 16)
m = Model()
