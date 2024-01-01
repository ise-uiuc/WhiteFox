
class Model(torch.nn.Module):
    def __init__(self, dropout_P=0.2):
        super().__init__()
 
class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
 
        self.query = torch.nn.Linear()
        self.value = torch.nn.Linear()
        self.key = torch.nn.Linear()
        self.dropout = torch.nn.Dropout(dropout_P)
 
    def forward(self, query, value, key, mask, use_layer_norm=True):
        q = self.query(query)
        v = self.value(value)
        k = self.key(key)
 
        b, n, w, d = q.shape
        assert b == 1, 'Batch size must be 1 to use multi head attention.'
        assert d == k.shape[-1] * self.heads,'d is not divisible by number of heads'
        b, m, w, d = v.shape
        assert m == 1, 'Batch size must be 1 to use multi head attention.'
        b, m, w, d = k.shape
        assert m == 1, 'Batch size must be 1 to use multi head attention.'
 
        # Reshape for multi-head attention
        q = q.reshape(1, n, self.heads, w, d)
        v = v.reshape(1, m, self.heads, w, d)
        k = k.reshape(1, m, self.heads, w, d)
 
        # Dot product
        qk = torch.matmul(q.transpose(-2, -1),k.transpose(-2, -1))
        s = qk.shape
        assert k.shape[-1] == d, 'd is not divisible by number of heads'
 
        # Scale
        inv_scale_factor = 1f / s[-1] ** 0.5
        scaled_qk = qk.div(inv_scale_factor)
 
        # Softmax
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
 
        # Dropout
        dropout_qk = self.dropout(softmax_qk)
 
        # Output
        output = torch.matmul(dropout_qk, v.transpose(-2, -1)).transpose(-3,-2).reshape(1, n, w, d)
        return output
    
# Initializing the model
m = MultiheadSelfAttention(heads)

# Inputs to the model
query = torch.randn(1, 3, 64, 16)
value = torch.randn(1, 1, 64, 64)
key = torch.randn(1, 1, 64, 64)
mask = None
