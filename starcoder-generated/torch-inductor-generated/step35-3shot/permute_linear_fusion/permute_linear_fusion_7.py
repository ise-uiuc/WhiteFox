
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = torch.unsqueeze(x1, dim=-1)
        v1 = self.linear(x2).clamp(min=0)
        x3 = torch.squeeze(v1, dim=-1)
        return torch.min(x3, dim=-1)[1]
# Inputs to the model
x1 = torch.randn(1, 2)
