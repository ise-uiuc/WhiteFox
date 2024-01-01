
class network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.nn.init.constant_(self.v1.weight, val=0.5)
    def forward(self, x):
        out = F.dropout(x)
        self.v1(out)
        return out
# Inputs to the model
x1 = torch.randn(32, 32)
