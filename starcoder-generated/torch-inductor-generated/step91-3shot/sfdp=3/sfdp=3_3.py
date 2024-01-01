
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, dim_q, dim_k, dim_v, dropout_p):
        super().__init__()
        self.n_head = n_head
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        if dim_k!= dim_v:
            raise ValueError('the dimension of kernal of key and value tensor must be equal!')
        self.dim_model = n_head * dim_v
        self.w_q = nn.Linear(dim_q, self.dim_model)
        self.w_k = nn.Linear(dim_k, self.dim_model)
        self.w_v = nn.Linear(dim_v, self.dim_model)
        self.dropout_layer = nn.Dropout(p=dropout_p)

        self.layer_norm = nn.LayerNorm(self.dim_q)
 
    def forward(self, query, key, value, attn_mask=None):
        residual = query
        batch_size = query.size(0)
        query = self.layer_norm(query)

        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        query_parts = torch.split(query, split_size_or_sections=self.dim_v, dim=-1)
        key_parts = torch.split(key, split_size_or_sections=self.dim_v, dim=-1)
        value_parts = torch.split(value, split_size_or_sections=self.dim_v, dim=-1)

        output_parts = []
        scale_factor = 1 / math.sqrt(self.dim_k)
        for i in range(self.n_head): 
            q = query_parts[i]
            k = key_parts[i]
            v = value_parts[i]
            qk = torch.matmul(q, k.transpose(-2, -1))
            scaled_qk = qk.mul(scale_factor)
            softmax_qk = scaled_qk.softmax(dim=-1)
            if attn_mask is not None:
                softmax_qk = softmax_qk * attn_mask
            dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_layer.p)
            output_part = dropout_qk.matmul(v)
            output_parts.append(output_part)

        output = torch.cat(output_parts, dim=-1).view(batch_size, -1, self.dim_model)
        output = output + residual
        return output

# Initializing the model
dim_query = dim_key = dim_value = 128
n_head = 4
dropout = 0.0
m = MultiHeadAttention(n_head, self.dim_query, self.dim_key, self.dim_value, dropout_p=0)

# Inputs to the model
x1 = torch.randn(1, 16, 128)
x2 = torch.randn(1, 16, 128)
x3 = torch.randn(1, 16, 128)
