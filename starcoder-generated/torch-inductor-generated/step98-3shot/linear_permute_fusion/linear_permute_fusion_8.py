
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()
    def forward(self, x0):
        v0 = x0
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = self.relu(v1)
        lstm1 = torch.nn.LSTM(2, 2)
        v3 = lstm1(v2)
        return v1
# Inputs to the model
x0 = torch.randn(1, 3, 2)
