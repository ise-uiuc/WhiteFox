
class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, n_dim_head=64, dropout_p=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_dim_head = n_dim_head
 
        self.key = torch.nn.Linear(dim, num_heads * n_dim_head)
        self.query = torch.nn.Linear(dim, num_heads * n_dim_head)
        self.value = torch.nn.Linear(dim, num_heads * n_dim_head)
 
        self.scale_factor = 1
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def _reshape(self, tensor, batch_size):
        tensor = tensor.reshape(batch_size, -1, self.num_heads, self.n_dim_head)
        return tensor.transpse(-2, -1)
 
    def forward(self, x1):
        batch_size = x1.shape[0]
        x2 = x1.transpose(-2, -1)
 
        key = self.key(x2)
        query = self.query(x2)
        value = self.value(x2)
 
        key = self._reshape(key, batch_size)
        query = self._reshape(query, batch_size)
        value = self._reshape(value, batch_size)
 
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
 
        output = output.reshape(batch_size, -1, self.num_heads * self.n_dim_head)
        return output

torch.manual_seed(0) # for stable result reproduction
att = Attention(dim=128)
att.scale_factor = 2 * (att.dim ** -0.5)
