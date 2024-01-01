
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(1,1,(1,2), (2,2))
    def forward(self, x):
        return self.layer(x)
x = torch.ones((1,1,3,2))
