
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.lstm = torch.nn.LSTM(2, 2)
    def forward(self, x0, x1):
        v0 = torch.nn.functional.linear(x0, self.linear.weight, self.linear.bias)
        v2 = torch.stack([v0, v0])
        v3 = torch.split(v2, 1, 1)
        v3b = [x.squeeze(1) for x in v3]
        if v3b[0].requires_grad:
            v3a = torch.cat(v3b)
            return v3a
        else:
            return v3b[0] + v3b[0]
# Inputs to the model
x0 = torch.randn(3, 1, 2)
x1 = torch.randn(3, 2, 2)
