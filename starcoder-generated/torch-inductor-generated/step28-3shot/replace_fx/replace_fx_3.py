
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, input):
        x = self.linear(input)
        y = torch.randint(low=2, high=6, size=(2, 2))
        res1 = torch.mul(x, y)
        z = torch.rand_like(x)
        res2 = torch.sum(x, dim=1, keepdim=True)
        res3 = torch.add(res2, z)
        return x
# Inputs to the model
input = torch.randn(1, 2, 2)
