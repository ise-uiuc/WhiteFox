
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)
 
    def forward(self, q, k, v, dropout):
        qk = torch.matmul(q, k.t())
        scale_factor = (2.0 / (torch.einsum('bhld,bhmd->bhlmd', [q, k])+1e-6)).to(dtype=q.dtype)
        softmax_qk = scale_factor.softmax(dim=-1)
        dropout_qk = softmax_qk * dropout
        output = dropout_qk.matmul(v)

        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 4, 1024)
k = torch.randn(2, 4, 1024)
v = torch.randn(2, 4, 1024)
dropout = torch.Tensor([0.5]).expand_as(q)
