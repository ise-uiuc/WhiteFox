
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout()
        self.sigmoid = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        x2 = x1.transpose(1, 2)
        v1 = torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
        v3 = self.gelu(v1)
        v4 = self.sigmoid(v3)
        v5 = self.dropout(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
