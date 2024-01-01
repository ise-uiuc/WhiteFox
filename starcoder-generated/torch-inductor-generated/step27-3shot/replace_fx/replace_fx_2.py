
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = x * (torch.nn.functional.dropout(x) + torch.nn.functional.dropout(x, p=0.3) + torch.nn.functional.dropout(x, p=0.4))
        return a1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
