
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(3, 8)
        self.k = torch.nn.Linear(3, 8)
        self.value = torch.nn.Linear(3, 3)
 
    def forward(self, x, b=10, p=0.5):
        v1 = self.q(x)
        v2 = self.k(x)
        v3 = torch.matmul(v1, v2.transpose(-2, -1))
        v4 = v3 * (self.state['key'].shape[-1] ** 0.5)
        v5 = torch.nn.functional.softmax(v4, dim=-1)
        v6 = torch.nn.functional.dropout(v5, p=p, training=True)
        return torch.matmul(v6, self.value(x))

# Inputs to the model
x = torch.randn(1, 3)

state = State(
    {'key': torch.randn(1, 3, 8)})

