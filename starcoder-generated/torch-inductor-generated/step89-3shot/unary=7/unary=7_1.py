
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = F.selu(x2)
        x4 = F.hardtanh(x3, min_val=0.0, max_val=6.0)
        x5 = x4 + 3
        x6 = x5 / 6
        return x6
l1 = torch.randn(1, 3);
