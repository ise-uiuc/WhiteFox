
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y=None):
        c1 = torch.nn.functional.dropout(x, p=0.2)
        return 1
# Inputs to the model
x1 = torch.randn(1)
