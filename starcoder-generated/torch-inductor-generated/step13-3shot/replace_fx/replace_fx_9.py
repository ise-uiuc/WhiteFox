
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        x1 = torch.nn.functional.dropout(x1, p=0.05)
        x2 = torch.rand_like(x2)
        x3 = torch.nn.functional.dropout(x3, p=float(x2.shape[1] < 8))
        return torch.sum(x1 + x2 + x3)
# Inputs to the model
x1 = torch.randn(5, 10, 5)
x2 = torch.randn(5, 20, 5)
x3 = torch.randn(20, 10)
