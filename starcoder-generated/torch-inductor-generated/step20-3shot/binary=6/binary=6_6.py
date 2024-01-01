
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
 
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 - torch.tensor([[1, -1, 3]], dtype=torch.float)
        return t2

