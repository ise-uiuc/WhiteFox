
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
 
    def forward(self, x):
        q = self.linear1(x1)
        k = self.linear1(x2)
        v = self.linear1(x3)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        scale_factor = torch.matmul(q,k.transpose(-2, -1))
        scale_factor = torch.softmax(scale_factor / math.sqrt(min(len(q[0]), len(k[0]))), dim=-1)
        drop_res = scale_factor * v
        return drop_res

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
x3 = torch.randn(1, 3)
