
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_transpose = torch.nn.Linear(12, 8, bias=False)
    def forward(self, x1):
        v1 = self.linear_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 12)
