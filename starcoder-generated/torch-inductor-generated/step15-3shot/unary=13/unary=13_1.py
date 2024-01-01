
class model_name(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(3, 3)

    def forward(self, x):
        x = self.l(x)
        x = F.sigmoid(x)
        x = x * x
        return x

# initialize the model.
m = model_name()

# inputs to the model.
x = torch.randn(1, 3)
