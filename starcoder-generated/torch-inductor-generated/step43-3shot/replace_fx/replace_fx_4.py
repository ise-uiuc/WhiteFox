
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.1)
        x3 = torch.rand_like(x1)
        x4 = torch.nn.functional.dropout(x3, p=0.1)
        x5 = torch.nn.functional.dropout(x1, p=0.2)
        x6 = torch.rand_like(x1)
        return x6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
