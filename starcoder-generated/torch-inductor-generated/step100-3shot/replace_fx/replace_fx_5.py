
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.nn.functional.dropout(x)
        t2 = F.dropout(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
