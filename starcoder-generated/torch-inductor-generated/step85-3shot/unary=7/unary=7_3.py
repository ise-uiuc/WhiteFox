
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 5),
            torch.nn.Linear(5, 2)
        )

    def forward(self, x1):
        out = self.model(x1)
        out_c = torch.clamp(out, min=0, max=6)
        out = out_c * 0.167
        return out * out_c
# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(1, 3)
