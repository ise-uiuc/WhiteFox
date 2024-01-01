
class Model(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.Q = torch.nn.Linear(dim, dim)
        self.K = torch.nn.Linear(dim, dim)
        self.V = torch.nn.Linear(dim, dim)
        self.dropout_p = 0.1
        self.scale_factor = (dim//num_heads)**-0.5
 
    def forward(self, query, key, value, mask):
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        if mask is not None:
            softmax_qk.masked_fill_(mask, value=float('-inf'))
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Input to the model
x1 = torch.randn(1, 32 * 3, 512)
x2 = torch.randn(1, 32 * 3, 512)
x3 = torch.randn(1, 32 * 3, 512)
mask = torch.ones(x1.shape[0], 32, 32)\
  .tril()


