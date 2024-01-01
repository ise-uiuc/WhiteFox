
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x1):
        a = self.linear(x1) + torch.tensor([2], device=x1.device)
        return F.dropout(a, p=0.5) * a
# Inputs to the model
x1 = torch.randn(1, 2, 4)
