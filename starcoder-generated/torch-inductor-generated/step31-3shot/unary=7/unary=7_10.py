
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, bias=False)
 
    def forward(self, x):
        v1 = self.act(x)
        v2 = v1 * torch.clamp(v1 + 3, 0, 6)
        return v2 / 6

