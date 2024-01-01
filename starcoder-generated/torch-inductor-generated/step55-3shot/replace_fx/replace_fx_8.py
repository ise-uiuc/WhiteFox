
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.rand_like(x)
        x = torch.nn.functional.dropout(x, p=1.0)
        x = torch.nn.functional.dropout(x, p=1.0)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
