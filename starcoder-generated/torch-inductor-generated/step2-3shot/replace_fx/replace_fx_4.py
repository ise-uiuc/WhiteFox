
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.3, inplace=True)
    def forward(self, x1):
        v1 = torch.nn.functional.dropout(x1, self.dropout.p)
        v2 = torch.randn_like(x1)
        return (v1 * v2).sum()
# Inputs to the model
x1 = torch.randn(1, 2, 2)
