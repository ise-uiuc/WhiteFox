
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x2 = torch.rand_like(x)
        x3 = torch.nn.functional.dropout(x, p=0.3)
        x4 = torch.nn.functional.dropout(x3, p=0.3)
        x5 = torch.nn.functional.dropout(x2, p=0.3)
        x6 = torch.rand_like(x4)
        x7 = torch.nn.functional.dropout(x6, p=0.96)
        x7 = torch.nn.functional.dropout(x7, p=0.96)
        return x7
# Inputs to the model
x = torch.randn(8, 6)
