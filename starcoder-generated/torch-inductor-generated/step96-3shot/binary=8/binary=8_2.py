
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTMCell(10, 10)
    def forward(self, x, h, c):
        v1 = self.lstm(x, (h, c))
        return v1[0]
# Inputs to the model
x = torch.randn(10, 10)
h = torch.randn(10, 10)
c = torch.randn(10, 10)
