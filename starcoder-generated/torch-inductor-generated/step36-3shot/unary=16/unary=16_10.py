
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear1 = torch.nn.Linear(20, 50)

    def forward(self, x1):
        v1 = self._linear1(x1)
        v2 = F.relu(v1)
        return v2

# Inputs to the model
x1 = torch.randn(4, 20)
