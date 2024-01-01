
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

other = torch.tensor([1.0, 2.0, 3.0])
x = torch.randn(1, 3)
