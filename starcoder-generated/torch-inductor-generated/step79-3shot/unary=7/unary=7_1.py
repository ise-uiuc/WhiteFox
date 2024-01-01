
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3)
 
    def forward(self, x):
        h1 = self.linear(x)
        h2 = h1 * torch.clamp(h1 + 3, min=0, max=6)
        h3 = h2 / 6
        return h3
m = Model()
