
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
    def forward(self, x):
        result = list()
        for _ in range(3):
            x = self.layer(x)
            result.append(x)
        x = torch.stack(result, dim=0)
        x = x.flatten(0, 1).flatten(1)
        return x
# Inputs to the model
x = torch.randn(2, 4)
