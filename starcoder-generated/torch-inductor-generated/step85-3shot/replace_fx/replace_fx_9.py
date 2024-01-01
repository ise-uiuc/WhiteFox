
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.dropout(x, p=0.88)
        x1 = F.dropout(x)
        x2 = torch.nn.functional.dropout(x1)
        x3 = F.dropout(x)
        x4 = torch.nn.functional.dropout(x3, p=0.777)
        x5 = torch.rand_like(x)
        z = F.dropout(x5,p=0.55)
        return x4
# Inputs to the model
x = torch.randn(1, 3)
