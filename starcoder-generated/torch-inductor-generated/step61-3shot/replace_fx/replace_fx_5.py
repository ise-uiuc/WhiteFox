
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, p=0.3)
        a2 = torch.nn.functional.dropout(x, p=0.05)
        return a1
# Inputs to the model
x = torch.randn(1)
