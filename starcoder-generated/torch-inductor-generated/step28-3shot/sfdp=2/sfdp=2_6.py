
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.hidden_size_per_head = self.hidden_size // self.num_heads
 
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x1):
        q = self.query(x1)
        k = self.key(x1)
        v = self.value(x1)
 
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, self.hidden_size_per_head).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, self.hidden_size_per_head).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, self.hidden_size_per_head).transpose(1, 2)
 
        dot_product = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1.0 / np.sqrt(self.hidden_size_per_head)
        scaled_dot_product = dot_product.div(inv_scale_factor)
 
        softmax_dot_product = scaled_dot_product.softmax(dim=-1)
        dropout_softmax_dot_product = self.dropout(softmax_dot_product)
 
        output = torch.matmul(dropout_softmax_dot_product, v)
        output = output.transpose(1, 2).reshape(output.shape[0], output.shape[1], -1)
 
        return output

# Initializing the model
m = Model(hidden_size=512, num_heads=4, dropout=0.)

# Inputs to the model
x1 = torch.randn(64, 4, 512)
