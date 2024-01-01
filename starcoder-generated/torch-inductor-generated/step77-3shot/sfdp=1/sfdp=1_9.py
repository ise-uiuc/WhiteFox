
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V):
        a1 = torch.matmul(Q, torch.transpose(K, -2, -1))
        a2 = a1 / 0.003086886698767729
        a3 = F.softmax(a2, dim=-1)
        a4 = F.dropout(a3, p=0.1)
        v1 = torch.matmul(a4, V)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
Q = torch.randn(5, 10, 384)
K = torch.randn(5, 20, 384)
V = torch.randn(5, 20, 384)
