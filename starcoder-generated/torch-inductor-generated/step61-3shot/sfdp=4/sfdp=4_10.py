
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.m = torch.nn.Softmax(dim=-1)
    def forward(self, query, value, key):
        Q = query @ key.transpose(-2, -1)
        A = torch.exp(Q) / math.sqrt(query.size(-1))
        B = torch.exp(Q)
        softmax = self.m(A.add(B)).mul(Q)
        output = softmax @ value
        return output
# Input to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
