
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = torch.nn.LSTM(1, 1, 1)
        self.gru2 = torch.nn.LSTM(1, 1, 1)
        self.gru3 = torch.nn.LSTM(1, 1, 1)
    def forward(self, x1, x2):
        y1 = self.gru1(x1)
        y2 = self.gru2(y1)
        y3 = self.gru3(y2)
        z1 = torch.rand_like(y3[0])
        return z1[0]
# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
