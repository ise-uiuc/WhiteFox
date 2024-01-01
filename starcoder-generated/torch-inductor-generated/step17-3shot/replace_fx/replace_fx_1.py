
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, a1):
        a2 = torch.rand_like(a1)
        a3 = torch.sum(torch.randn_like(a2))
        a4 = torch.nn.functional.dropout(torch.sum(a2))
        return a3, a4
# Inputs to the model
x1 = torch.randn(5, 7, 7)
