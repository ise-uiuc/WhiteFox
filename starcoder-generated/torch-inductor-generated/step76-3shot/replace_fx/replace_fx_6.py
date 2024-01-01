
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        d1 = torch.rand(()).item()
        d2 = torch.rand(())
        x = F.dropout(x, p=0.5)
        x = torch.rand(())
        return x + d1
# Inputs to the model
x = torch.randn(1, 1, 2)
