
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(7, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        