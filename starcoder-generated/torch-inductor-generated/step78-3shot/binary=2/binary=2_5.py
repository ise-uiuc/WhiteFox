
# Here you're implementing an LSTM based model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTMCell(16, 16)
    def forward(self, x, h, c):
        v = self.lstm(x, (h, c))
        return v - x[:, 1]
# Inputs to the model
x = torch.randn(10, 16)
h = torch.randn(10., 16)
c = torch.randn(10., 16)
