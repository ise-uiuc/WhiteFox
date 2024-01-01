
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight)
        x2 = torch.tensor([[0., 0., 0., 1.], [0., 0., 0., 1.], [0., 0., 0., 1.]], dtype=torch.float)
        return (v2 + x2)
# Inputs to the model
x1 = torch.randn(1, 3, 2)
