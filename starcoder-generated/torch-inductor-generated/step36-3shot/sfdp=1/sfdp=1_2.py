
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(64, 128)
        self.key = torch.nn.Linear(64, 128)
        self.value = torch.nn.Linear(64, 128)
 
    def forward(self, Q, K, V):
        v1 = self.query(Q)
        v2 = self.key(K)
        v3 = self.value(V)
        v4 = torch.matmul(v1, v2.transpose(-2, -1))
        v5 = v4.div(16)
        v6 = torch.nn.functional.softmax(v5, dim=-1)
        v7 = 0.1 * v6
        v8 = torch.matmul(v7, v3)
        return v8

# Initializing the model
m = Model()

device = torch.device("cpu")
x1 = torch.randn(1, 64, device=device)
x2 = torch.randn(1, 64, device=device)
x3 = torch.randn(1, 64, device=device)

__input__ = [x1, x2, x3]
