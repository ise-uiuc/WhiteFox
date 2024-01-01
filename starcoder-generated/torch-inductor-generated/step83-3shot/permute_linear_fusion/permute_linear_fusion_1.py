
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.elu = torch.nn.ELU(alpha=0, inplace=True)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight)
        v3 = self.elu(v2) + torch.nn.functional.relu(self.linear1(v2))
        v4 = v2.detach()
        return self.elu(v2) + v4.sum(dim=-1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
