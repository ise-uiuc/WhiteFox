
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.nn.Dropout(p=0.3, inplace=True)
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.3)
        a2 = torch.rand_like(x1)
        return (a1 + a2).mean()
# Inputs to the model
x1 = torch.randn(1, 2)
