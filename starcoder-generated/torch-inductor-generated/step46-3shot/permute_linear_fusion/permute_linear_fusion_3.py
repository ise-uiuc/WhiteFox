
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout()
        self.sigmoid = torch.nn.Sigmoid()
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.sigmoid(v2)
        v4 = self.dropout(v3)
        v5 = self.gelu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
