
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(2, 4),
            torch.nn.ReLU()
        )
    def forward(self, x2):
        v2 = x2
        v3 = self.layer0(v2)
        v3 = v3.permute(0, 2, 1)
        return v3
# Inputs to the model
x2 = torch.randn(1, 2, 2)
