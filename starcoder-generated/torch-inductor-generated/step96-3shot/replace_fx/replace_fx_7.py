
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 + 1.0
        x1 = torch.nn.functional.dropout(x1)
        x2 = torch.rand_like(x1, dtype=torch.float32)
        x3 = torch.unsqueeze(x2, dim=0)
        x4 = torch.rand_like(x3, dtype=torch.float32)
        return x4 + x1
# Inputs to the model
x1 = torch.randn(2)
