
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x, p=0.5)
        x2 = torch.rand_like(x1, requires_grad=True)
        return x1
# Inputs to the model
x1 = torch.randn(10, 4)
