
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.ReLU6()
        self.tanh = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        v4 = self.tanh(v3)
        v5 = self.dropout(v4)
        return torch.squeeze(v5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
