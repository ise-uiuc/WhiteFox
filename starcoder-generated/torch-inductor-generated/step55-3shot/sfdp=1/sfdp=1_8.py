
class Model(torch.nn.Module):
    def __init__(self, num_heads=8):
        super().__init__()
        # The weight tensor of the linear transformation from query tensor to query key weights
        self.query_weight = torch.nn.Parameter(torch.empty(num_heads, E_DIM, KEY_DIM))
        # The bias tensor of the linear transformation from query tensor to query key weights
        self.query_bias = torch.nn.Parameter(torch.empty(num_heads, E_DIM))
        # The weight tensor of the linear transformation from key tensor to query key weights
        self.key_weight = torch.nn.Parameter(torch.empty(num_heads, E_DIM, KEY_DIM))
        # The bias tensor of the linear transformation from key tensor to query key weights
        self.key_bias = torch.nn.Parameter(torch.empty(num_heads, E_DIM))
        # The weight tensor of the linear transformation from value tensor to query key weights
        self.value_weight = torch.nn.Parameter(torch.empty(num_heads, E_DIM, VALUE_DIM))
        # The bias tensor of the linear transformation from value tensor to query key weights
        self.value_bias = torch.nn.Parameter(torch.empty(num_heads, E_DIM))
        # The weight tensor of the linear transformation from input encoding to query key weights
        self.encdec_weight = torch.nn.Parameter(torch.empty(num_heads, E_DIM, VALUE_DIM))
        # The bias tensor of the linear transformation from input encoding to query key weights
        self.encdec_bias = torch.nn.Parameter(torch.empty(num_heads, E_DIM))
        # The output dimension of the query key weights
        self.num_heads = num_heads
        # The inverse scale factor
        self.inv_scale_factor = 1.0 / math.sqrt(VALUE_DIM)
        # The dropout rate
        self.dropout_p = 0.1

    def forward(self, query, key, value, encdec, pos_emb):
        q = torch.matmul(query, self.query_weight) + self.query_bias # Linear transformation of the query tensor
        k = torch.matmul(key, self.key_weight) + self.key_bias # Linear transformation of the key tensor
        v = torch.matmul(value, self.value_weight) + self.value_bias # Linear transformation of the value tensor
        e = torch.matmul(encdec, self.encdec_weight) + self.encdec_bias # Linear transformation of the input encoding
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of q and k
        scaled_qk = qk.div(self.inv_scale_factor) # Scale the dot product by the inverse scale factor
        pos_emb = pos_emb.div(self.inv_scale_factor) # Scale the positional embedding by the inverse scale factor
        scaled_qk += pos_emb.unsqueeze(-2) # Add the positional embedding
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) + e
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, T_SEQ, E_DIM)
key = torch.randn(1, T_SEQ, E_DIM)
value = torch.randn(1, T_SEQ, E_DIM)
encdec = torch.randn(1, T_SEQ, E_DIM)
pos_emb = torch.randn(1, T_SEQ, E_DIM)
