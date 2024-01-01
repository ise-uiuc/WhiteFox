
class m1(torch.nn.Module):
    def __init__(self, m2):
        super().__init__()
    def forward(self, x1):
        x2 = x1
        x3 = torch.nn.functional.dropout(x1)
        x4 = self.m2(x2)
        return torch.nn.functional.dropout(x3)
class m2(torch.nn.Module):
    def forward(self, x1):
        x2 = x1
        x3 = torch.nn.functional.dropout(x2)
        return x3        
# Inputs to the model
x1 = torch.randn(1, 2, 2)
