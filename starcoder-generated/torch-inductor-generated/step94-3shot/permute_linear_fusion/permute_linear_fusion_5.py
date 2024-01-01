
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout2d()
        self.sigmoid = torch.nn.PReLU()
        self.gelu = torch.nn.RNN(2, 2, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        v4 = v3.view(v3.size(0), 1, 2)
        v5 = self.dropout(v4)
        v6 = self.gelu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
