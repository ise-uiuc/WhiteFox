
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.query = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.value = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        y1 = self.query(x1)
        y2 = self.key(x1)
        y3 = y2.transpose(-2, -1)
        y4 = torch.matmul(y1, y3)
        y5 = y4 / math.sqrt(8)
        y6 = y3 + 1
        v1 = torch.softmax(y6, dim=1)
        y7 = self.value(x1)
        v2 = v1 @ y7
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
