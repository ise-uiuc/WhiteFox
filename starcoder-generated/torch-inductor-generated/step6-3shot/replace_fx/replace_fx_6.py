
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2)
    def forward(self, x):
        v = torch.nn.functional.dropout(x, p=0.2)
        w = torch.nn.Softplus(beta=1.2, threshold=2.2)(1.7)
        z1 = torch.nn.functional.dropout(w, p=0.0)
        z2 = torch.nn.functional.dropout(w, p=0.0)
        z3 = torch.nn.functional.dropout(w, p=0.0)
        z4 = torch.nn.functional.dropout(w, p=0.0)
        return torch.nn.functional.dropout(x)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
