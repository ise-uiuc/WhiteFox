
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout()
        self.sigmoid = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.dropout(v2, 0.0, True)
        v4 = torch.nn.functional.gelu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
