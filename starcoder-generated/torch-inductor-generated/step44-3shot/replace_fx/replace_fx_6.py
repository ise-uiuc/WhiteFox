
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
    def forward(self, x1):
        a = torch.rand_like(x1) + torch.tensor([2], device=x1.device)
        x2 = a * torch.nn.functional.dropout(self.linear(a))
        a = torch.randn((), device=x1.device)
        return F.dropout(x2, p=0.5, training=False)
# Inputs to the model
x1 = torch.randn(1, 2, 4)
