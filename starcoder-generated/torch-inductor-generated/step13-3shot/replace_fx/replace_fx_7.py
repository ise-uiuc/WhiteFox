
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 2)

    def forward(self, x1):
        t1 = self.layer1(x1)
        x1 = torch.rand((t1.shape[0], 4), device="cuda")
        x1 = self.layer1(x1)
        return x1
# Inputs to the model
x1 = torch.rand(1, 10)
