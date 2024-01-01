
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4, 5, bias=False)
 
    def forward(self, x):
        v = self.linear(x)
        