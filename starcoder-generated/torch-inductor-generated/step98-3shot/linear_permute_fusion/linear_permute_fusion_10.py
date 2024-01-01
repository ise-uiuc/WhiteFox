
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm0 = torch.nn.LSTM(2, 2)
        self.lstm1 = torch.nn.LSTM(2, 2)
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v0 = x1
        v1 = self.lstm0(v0)
        v2 = v1.permute(0, 3, 1, 2)
        v3 = self.lstm1(v2)
        v5 = self.linear(v3)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 2)
