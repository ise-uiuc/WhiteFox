
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x3):
        v2 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        v3 = v2.permute(0, 2, 1)
        self.lstm = torch.nn.LSTM(2, 2)
        lstm = self.lstm(v3)[0]
        return self.linear(lstm)
# Inputs to the model
x3 = torch.randn(1, 3, 2)
