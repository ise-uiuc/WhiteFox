
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(300, 500)
        self.linear2 = torch.nn.Linear(500, 4)
    def forward(self, x):
        y1 = self.linear(x)
        y2 = y1  * 0.5
        y3 = y1 * 0.7071067811865476
        y4 = torch.erf(y3)
        y5 = y4 + 1
        y6 = y2 * y5
        y7 = self.linear2(y6)
        return y7
# Inputs to the model
x = torch.randn(1, 300)
