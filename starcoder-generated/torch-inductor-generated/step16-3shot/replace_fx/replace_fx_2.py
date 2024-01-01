
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        xb = x1.clone(memory_format=torch.channels_last)
        a1 = torch.nn.functional.dropout(xb, p=0.5)
        a2 = torch.rand_like(x1)
        return torch.rand_like(a1)
# Inputs to the model
x1 = torch.randn(2, 2, 2, 2)
