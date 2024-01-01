
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
 
    def forward(self, x0, other=torch.Tensor([1, 2, 3, 4])):
        v0 = self.linear(x0)
        v1 = v0 + other
        v2 = torch.relu(v1)
        v3 = torch.erf(v2)
        return v3, v2

