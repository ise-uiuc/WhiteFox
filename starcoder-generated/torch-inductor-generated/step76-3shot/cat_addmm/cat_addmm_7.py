
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, t1):
        t2 = self.layers(t1)
        t3 = torch.stack((t2, t2, t2))
        t3 = torch.flatten(t3, 1)
        return t3
# Inputs to the model
t1 = torch.randn(2, 2)
