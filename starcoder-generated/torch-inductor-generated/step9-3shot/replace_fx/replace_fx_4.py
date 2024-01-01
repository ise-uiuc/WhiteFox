
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = torch.nn.modules.dropout(p=0.1)
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.9)
        u = x2.uniform_(0, 1)
        v = x1.size(3)
        x3 = torch.nn.functional.dropout(x1, p=0.2)
        w = self.d1(x2)
        return x3
# Inputs to the model
x1 = torch.randn(3, 3, 3)
