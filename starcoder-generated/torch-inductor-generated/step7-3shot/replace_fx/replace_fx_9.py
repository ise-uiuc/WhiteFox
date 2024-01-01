
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, self.dropout.p)
        a2 = torch.randn_like(x1)
        a3 = torch.sum(torch.randn_like(torch.abs(a2)))
        a4 = torch.nn.functional.dropout(torch.sum(a2))
        return a3
# Inputs to the model
x1 = torch.randn(1, 10)
