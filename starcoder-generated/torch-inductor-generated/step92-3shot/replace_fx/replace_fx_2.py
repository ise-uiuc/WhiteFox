
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x, n):
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.rand_like(x, n)
        x = torch.nn.functional.relu6(x)
        x = x * 0.2 + 0.5
        return F.dropout(x, p=0.1) + 1
# Inputs to the model
n = torch.randn([2, 2, 2])
x = torch.randn(2, 2, 2)
