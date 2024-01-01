
class Model(torch.nn.Module):
    def __init__(self, dim_size, num_heads):
        super().__init__()
        self.key = torch.nn.Linear(dim_size, dim_size)
        self.query = torch.nn.Linear(dim_size, dim_size)
        self.value = torch.nn.Linear(dim_size, dim_size)

    def forward(self, x1):
        k = self.key(x1)
        q = self.query(x1)
        v = self.value(x1)
        qk = torch.matmul(k, q.transpose(-2, -1))
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output, mask

# Initializing the model
m = Model(128, 4)

# Inputs to the model
x1 = torch.randn(4, 128)
mask = torch.zeros(4, 4)
__output__, __mask__ = m(x1, mask)
# mask: output = softmax(Q * K) V

