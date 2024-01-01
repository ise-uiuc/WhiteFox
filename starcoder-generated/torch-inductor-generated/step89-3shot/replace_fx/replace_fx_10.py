
class m1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 5)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(5, 6)
    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.relu1(y1)
        y3 = self.dropout1(y2)
        y4 = self.linear2(y3)
        y5 = torch.rand_like(y4)
        y6 = y5 - y4
        return y6
# Inputs to the model
x1 = torch.randn(1, 3, 3)
