
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.rand_like(x)
        x = torch.nn.functional.dropout(x, p=0.2)
        return x
# Inputs to the model
x1 = torch.randn(1)
