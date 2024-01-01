
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm((2, 2), 1e-05, 0.125)
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout()
    def forward(self, x1):
        v1 = self.layernorm(x1)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = v3.transpose(1, 2)
        v5 = self.dropout(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
