
class m2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=16, out_features=16, bias=True)
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = torch.nn.functional.dropout(x2, p=0.1)
        x4 = torch.rand_like(x3)
        return torch.nn.functional.dropout(x4)
# Inputs to the model
x1 = torch.randn(1, 2, 16)
