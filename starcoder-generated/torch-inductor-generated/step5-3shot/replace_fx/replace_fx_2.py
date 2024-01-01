
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.23)
        a2 = torch.nn.functional.dropout(x1, p=0)
        a3 = a1 * a2
        return a2
# Inputs to the model
x1 = torch.randn(1, 2)
