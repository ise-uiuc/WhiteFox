
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other
    
    def forward(self, x1):
        v1 = F.linear(x1, self.other)
        return v1