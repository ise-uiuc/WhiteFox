
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.dropout = torch.nn.Dropout(p=0)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(0)
        v3 = torch.cat([v3 for _ in range(3)], dim=0)
        v3 = v3.unsqueeze(dim=0)
        v3 = v3.permute(0, 2, 1, 3, 4)
        v3 = v3.squeeze(0)
        v3 = self.dropout(v3)
        v4 = v2
        return self.dropout(v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
