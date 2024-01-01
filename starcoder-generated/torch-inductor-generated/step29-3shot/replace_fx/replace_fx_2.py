
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.nn.functional.dropout(x1, p=0.4)
        u1 = torch.rand_like(x2) + \
        torch.ones_like(x2) * torch.rand_like(x2) # There are multiple occurrences
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 3)
x2 = torch.randn(1, 3, 3)
