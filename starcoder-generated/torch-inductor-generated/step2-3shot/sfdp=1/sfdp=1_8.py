
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, n_heads, dropout_p=0.5):
        super().__init__()
        self.q_proj = torch.nn.Linear(q_dim, n_heads * k_dim)
        self.k_proj = torch.nn.Linear(k_dim, n_heads * k_dim)
        self.v_proj = torch.nn.Linear(v_dim, n_heads * v_dim)
        self.n_heads = n_heads
        self.dropout = torch.nn.Dropout(dropout_p)
     
    def forward(self, query, key, value, inv_scale_factor):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = self._split_heads(q, self.n_heads)
        k = self._split_heads(k, self.n_heads)
        v = self._split_heads(v, self.n_heads)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1).div(inv_scale_factor))
        softmax = torch.nn.Softmax(dim=-1)
        softmax_qk = softmax(scaled_qk)
        softmax_qk = self.dropout(softmax_qk)
        output = torch.matmul(softmax_qk, v)
        output = self._combine_heads(output)
        return output
    
    def _split_heads(self, tensor, n_heads):
        batch_size = tensor.shape[0]
        size, rem = divmod(tensor.shape[-1], n_heads)
        assert rem == 0, "{} is not divisible by {}".format(tensor.shape[-1], n_heads)
        shape = tuple(list(tensor.shape)[:-1]) + (n_heads, size)
        tensor = tensor.reshape(shape)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size, -1, size)
        return tensor

    def _combine_heads(self, tensor):
        batch_size = tensor.shape[0]
        n_heads, size = tensor.shape[1], tensor.shape[-1]
        shape = tuple(list(tensor.shape)[:-2]) + (n_heads * size,)
        tensor = tensor.reshape(shape)
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size, -1, size)
        return tensor

# Initializing the multihead attention module
def init_multihead(q_dim, k_dim, v_dim, n_heads):
    m = MultiHeadAttention(q_dim, k_dim, v_dim, n_heads)
    x1 = torch.randn(1, 3, 64, 64)
    x2 = torch.randn(1, 4, 64, 64)
    x3 = torch.randn(1, 4, 64, 64)
    