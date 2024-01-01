
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.zeros_like(x).repeat(3, 4).to(x.device)
        x = x.permute(1, 0).view(4, -1).transpose(1, 0)
        x = x.permute(1, 0)
        return x
# Inputs to the model
x = torch.randn(2, 3)
