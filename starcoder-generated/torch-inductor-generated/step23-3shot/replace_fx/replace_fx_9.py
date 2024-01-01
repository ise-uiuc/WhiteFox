
class m1(torch.nn.Module):
    def __init__(self,p1,p2):
        super().__init__()
        self.p1 = p1
        self.p2 = p2
    def forward(self, x1):
        x2 = self.p1(x1)
        x3 = self.p2(x2)
        x4 = torch.randint(0, 1, (1,))
        x5 = torch.rand_like(x4)
        x6 = torch.nn.functional.dropout(x3, p=0.2)
        x7 = torch.nn.functional.relu(x6)
        x8 = torch.abs(x7)
        x9 = torch.nn.functional.linear(x8, x5, bias=42)
        x10 = torch.relu(x6)
        return x9
p1 = nn.Linear(2, 4)
p2 = nn.Dropout()
x = torch.randn(1, 2, 2)
