
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2):
        a2 = torch.randn_like(x2)
        a3 = torch.abs(x2 - torch.nn.functional.dropout(x2))
        v1 = torch.tensor(v2)
        return x2
# Inputs to the model
x2 = torch.randn(1, 1)
