
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.dropout(x1, p=0.8)
        v2 = torch.rand_like(x1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, requires_grad = True)
