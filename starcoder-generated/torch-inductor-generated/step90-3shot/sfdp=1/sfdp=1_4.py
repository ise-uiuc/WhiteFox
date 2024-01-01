
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = torch.nn.Linear(2, 4)
        self.proj2 = torch.nn.Linear(2, 4)
 
    def forward(self, q, k):
        q1 = self.proj1(q)
        q2 = self.proj2(q)
        k1 = self.proj1(k)
        k2 = self.proj2(k)
        qk1 = torch.matmul(q1, k1.transpose(-2, -1))
        qk2 = torch.matmul(q2, k2.t())
        qk = (qk1 + qk2) / 2
        return qk

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 2)
k = torch.randn(5, 2)
