
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 8)
        self.linear2 = torch.nn.Linear(8, 4)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 5
        v3 = F.relu(v2)
        return self.linear2(v3)
# Inputs to the model
x1 = torch.randn(1, 5) # Input tensor x1 should be of shape (N, M) where N and M are integers satisfying: 1 <= N, M <= 5
