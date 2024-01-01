
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16, bias=True)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        _min = 0
        _max = 6
        l2 = l1 * torch.clamp(l1 + 3, min=_min, max=_max)
        l3 = l2 / 6
        return l3

# Input to the model
x1 = torch.randn(1, 64, requires_grad=True)
m = Model()
y = m(x1)

y.backward()
x1_grad = x1.grad.clone()

