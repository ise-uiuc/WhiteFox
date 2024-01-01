
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3, out_features=4, bias=False)
        self.linear.weight = torch.nn.parameter.Parameter(torch.ones_like(self.linear.weight))
    def forward(self, x1):
        v1 = torch.nn.functional.dropout(x1, p=(0.2))
        v2 = torch.rand_like(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3)
