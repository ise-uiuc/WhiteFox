
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(2, 3)
    def forward(self, x0):
        v3 = self.lstm(x0)[1][0]
        v4 = self.lstm(x0)[-1]
        v5 = self.lstm[0](self.lstm[1:3](self.lstm[4](x0)))
        return v5
# Inputs to the model
x0 = torch.randn(1, 3, 2)
