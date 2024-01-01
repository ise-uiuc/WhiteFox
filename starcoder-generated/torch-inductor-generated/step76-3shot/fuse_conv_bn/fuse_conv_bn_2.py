
class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x1):
        x1 = torch.add(torch.add(x1, x1), x1)
        return torch.add(x1, x1)
# Inputs to the model
x1 = torch.randn(1,3,4,4)
