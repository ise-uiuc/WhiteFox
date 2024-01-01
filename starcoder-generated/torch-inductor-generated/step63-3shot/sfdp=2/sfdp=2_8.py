
class Model(torch.nn.Module):
    def __init__(self, n_head=16, d_head=32, dropout_p=0.0):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.dk = d_head // n_head

        # Linear projection for q(queries), k(keys), and v(values) from d_in to d_k * n_head
        self.query_proj = torch.nn.Linear(d_in, d_k * n_head)
        self.key_proj = torch.nn.Linear(d_in, d_k * n_head)
        self.value_proj = torch.nn.Linear(d_in, d_k * n_head)

        self.dropout = torch.nn.Dropout(dropout_p)

        # Affine transform for multi-head
        self.layer_norm = torch.nn.LayerNorm(d_in)
        self.output_proj = torch.nn.Linear(d_in, d_in)
        self.scale_factor = math.sqrt(d_head)

    def forward(self, query, key, value):
        batch_size, q_len, d_in = query.size()
        d_k, n_head = self.dk, self.n_head

        q = self.query_proj(query).view(batch_size, q_len, n_head, d_k)
        k = self.key_proj(key).view(batch_size, k_len, n_head, d_k)
        v = self.value_proj(value).view(batch_size, k_len, n_head, d_k)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        q = q.contiguous().view(batch_size * n_head, q_len, d_k)
        k = k.contiguous().view(batch_size * n_head, k_len, d_k)
        v = v.contiguous().view(batch_size * n_head, k_len, d_k)

        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / self.scale_factor

        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        
        output = torch.matmul(softmax_qk, v)
        output = output.view(batch_size, n_head, q_len, d_k)
        output = output.transpose(1,2).contiguous()

        # output = [batch_size, q_len, out_dim]
        output = output.view(batch_size, q_len, d_in)
        output = self.dropout(output)
        output = self.layer_norm(query + output)
        output = self.output_proj(output)
        # output = [batch_size, q_len, out_dim]
        return output

# Initializing the model
m = Model()

# Shape of query, key, and value
query = torch.randn(16, 32, 128)
key = torch.randn(16, 64, 128)
value = torch.randn(16, 64, 128)
output = m(query, key, value)

