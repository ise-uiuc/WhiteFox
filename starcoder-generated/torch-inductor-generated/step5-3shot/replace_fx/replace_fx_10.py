
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.5)
        return torch.nn.functional.dropout(a1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
