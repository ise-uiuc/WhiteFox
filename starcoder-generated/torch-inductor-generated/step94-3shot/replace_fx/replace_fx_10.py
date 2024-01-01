
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.2)
        x3 = torch.nn.functional.dropout(x2, p=0.2, training=True)
        x4 = torch.rand_like(x3)
        x5 = torch.unsqueeze(x4, dim=0)
        x6 = torch.rand_like(x5)
        return x4 + x6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
