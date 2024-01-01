
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v1 = v2 - __other__
        return v1

