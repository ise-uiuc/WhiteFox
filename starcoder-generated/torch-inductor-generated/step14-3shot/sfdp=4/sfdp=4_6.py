
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(64, 32)
        self.key = torch.nn.Linear(32, 16)
        self.weight = torch.tensor([32/32/32, 16/32/16])
        self.weight.requires_grad = False
        self.value = torch.nn.Linear(32, 32)
        self.mask = torch.tensor([[0], [0], [0], [1], [1]], dtype=torch.float32)
 
    def forward(self, x1):
        q1 = self.query(x1)
        k1 = self.key(q1)
        qk1 = self.weight.view(-1) * q1 * k1
        p = qk1 + self.mask
        return torch.softmax(p, dim=-1) @ self.value(q1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 32)
