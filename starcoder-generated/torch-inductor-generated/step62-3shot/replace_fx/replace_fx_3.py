
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.dropout(x, p=0.5)
        x = torch.rand_like(x)
        x = torch.nn.functional.dropout(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
