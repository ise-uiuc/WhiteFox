
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.2)
        x3 = torch.rand_like(x4)
        x3 = torch.rand_like(x3)
        x4 = torch.nn.functional.dropout(x2 + x3, p=0.2)
        return x4
# Inputs to the model
x1 = torch.arange(768)
