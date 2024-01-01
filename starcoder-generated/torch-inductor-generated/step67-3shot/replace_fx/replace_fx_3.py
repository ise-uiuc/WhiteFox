
class test_dropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        c1 = torch.nn.functional.dropout(x)
        c2 = torch.rand_like(x)
        return c1
# Inputs to the model
x = torch.randn(2)
