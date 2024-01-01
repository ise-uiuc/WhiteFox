
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1).unsqueeze(-1)
        v3 = v2.squeeze(0)
        lstm2 = torch.nn.LSTM(1, 1)
        return lstm2(v3)[0]
# Inputs to the model
x1 = torch.randn(1, 3, 2)
