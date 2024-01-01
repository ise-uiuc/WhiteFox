
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(100, 10)
    def forward(self, x):
        x = self.layers(x)
        x = x.softmax(dim=-1)
        f = x[:,3].unsqueeze(-1)
        e = x[:,4].unsqueeze(-1)
        x = x.gather(dim=-1, index=f)
        x = x / e
        x = x.log()
        return x
# Inputs to the model
x = torch.randn(5, 100)
