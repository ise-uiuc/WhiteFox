
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        lstm1 = torch.nn.LSTM(1, 3, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(6, 1)
    def forward(self, x3):
        v3_1 = lstm1(x3)
        v3_2 = v3_1[0]
        v3_2_re = v3_2.repeat(1, 1, 1)
        v2 = torch.nn.functional.linear(v3_2, self.linear.weight, self.linear.bias)
        lstm3 = torch.nn.LSTM(1, 3, bidirectional=True, batch_first=True)
        v4 = lstm3(v2.permute(1, 0, 2))
        v5 = torch.nn.functional.linear(v4[0], self.linear.weight, self.linear.bias)
        return v5
# Inputs to the model
x3 = torch.randn(1, 3, 1)
