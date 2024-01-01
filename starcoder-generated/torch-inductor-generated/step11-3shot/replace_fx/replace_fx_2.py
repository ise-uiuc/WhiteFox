
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, p=0.2, training=False)
        a2 = torch.nn.functional.dropout(x, p=0.3, training=False)
        a3 = torch.nn.functional.dropout(x)
        c1 = a1 + a2 + a3
        return 1
# Inputs to the model
x1 = torch.randn(1)
