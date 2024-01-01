
class ReluLinearMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, False)

    def forward(self, x2):
        v1 = self.linear(x2)
        if not isinstance(v1, torch.Tensor):
            return [v1]
        v1 = F.relu(v1, inplace=True)
        return v1

# Initializing the model
m = ReluLinearMod()

# Input to the model
x2 = torch.randn(1, 16)
