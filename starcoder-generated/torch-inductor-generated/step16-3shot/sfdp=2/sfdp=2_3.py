
class Model(torch.nn.Module):
    def __init__(self, hidden, num_hiddens, dropout_prob):
        super().__init__()
        self.fc_scale_factor = torch.nn.Linear(hidden, num_hiddens, bias=False)
        self.fc_inv_scale_factor = torch.nn.Linear(hidden, num_hiddens, bias=False)
        self.dropout = torch.nn.Dropout(dropout_prob)
 
    def forward(self, query, key, value, attn_mask):
        scale_factor = self.fc_scale_factor(query)
        inv_scale_factor = torch.log1p(torch.exp(self.fc_inv_scale_factor(query)))
 
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
 
        self.head_size = hidden_size // num_heads
        self.scale = self.head_size ** -0.5
        self.dropout_prob = dropout
         
        self.projection_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.projection_k = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.projection_v = torch.nn.Linear(hidden_size, hidden_size, bias=False)
 
        self.fc = torch.nn.Linear(hidden_size, hidden_size, bias=False)
 
    def forward(self, query, key, value, attn_mask):
        query = self.projection_q(query)
        key = self.projection_k(key)
        value = self.projection_v(value)
         
        k_input_size = (query.size(0), -1, self.num_heads, self.head_size)
        q_input_size = (key.size(0), -1, self.num_heads, self.head_size)
        v_input_size = (value.size(0), -1, self.num_heads, self.head_size)
 
        q = self._reshape_to_batches(query, q_input_size)
        k = self._reshape_to_batches(key, k_input_size)
        v = self._reshape_to_batches(value, v_input_size)
 
        outputs = self._attention(q, k, v, (query.size(1), key.size(1)))
        outputs = self._reshape_from_batches(outputs, (query.size(0), -1, self.hidden_size))
 
        output = self.fc(outputs)
 
        return output
 
    def _attention(self, query, key, value, mask):
        output = (query * self.scale) @ key
         
        output -= 1e30 * (1 - mask[0])[:, None]
        output = torch.nn.functional.softmax(output, dim=-1)
        output = self.dropout(output)
 
        output = output @ value
        return output
 
    def _reshape_to_batches(self, x, new_size):
        new_shape = (x.size(0) // new_size[0], new_size[0]) + new_size[1:]
        x = x.view(new_shape)
        x = x.transpose(0, 1)
        return x.reshape(-1, *new_size[2:])
 
    def _reshape_from_batches(self, x, old_size):
        old_shape = (-1, old_size[0], self.num_heads, self.head_size)
        x = x.reshape(old_shape)
        x = x.transpose(0, 1)
        new_shape = (old_size[0], -1) + old_shape[2:]
        return x.reshape(*new_shape)

# Initializing the model
hidden_size = 1024 
num_hiddens = hidden_size * 4
dropout_p = 0.1
num_heads = 4 
m = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout_p)

# Inputs to the model
query = torch.randn(1, 100, hidden_size)
key = torch.randn(1, 1000, hidden_size)
value= torch.randn(1, 1000, hidden_size)
attn_mask = torch.ones(1, query.shape[1], key.shape[1])  # (batch_size, query_seq_length, key_seq_length)

