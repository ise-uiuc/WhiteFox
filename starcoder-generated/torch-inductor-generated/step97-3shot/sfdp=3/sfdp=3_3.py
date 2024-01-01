
class Model(torch.nn.Module):
    def __init__(self, query_vector_dim, key_vector_dim, value_vector_dim, num_heads,
                 scale_factor=None, dropout_p=0):
        super().__init__()
        self.query_projection = torch.nn.Linear(query_vector_dim, num_heads * key_vector_dim)
        self.key_projection = torch.nn.Linear(key_vector_dim, num_heads * key_vector_dim)
        self.value_projection = torch.nn.Linear(value_vector_dim, num_heads * value_vector_dim)
        self.num_heads = num_heads
        self.scale_factor = torch.nn.Parameter(torch.tensor(scale_factor, dtype=torch.float32)) if scale_factor else None
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        q_projection = self.query_projection(query)
        k_projection = self.key_projection(key).transpose(1, 2)
        v_projection = self.value_projection(value)
        q_shape = q_projection.size()
        v_shape = v_projection.size()
        qk = torch.matmul(q_projection, k_projection).view(q_shape[:-1] + (q_shape[-1], v_shape[-1]))
        scaled_qk = torch.matmul(qk, self.scale_factor) if self.scale_factor else qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, v_projection).view(q_shape[:-1] + (v_shape[-1],))
        return output

query_vector_dim, key_vector_dim, value_vector_dim = 128, 128, 128
num_heads = 4
scale_factor, dropout_p = None, 0.5

# Initializing the model
m = Model(query_vector_dim, key_vector_dim, value_vector_dim, num_heads,
          scale_factor, dropout_p)

# Inputs to the model
query = torch.randn(1, 8, query_vector_dim)
key = torch.randn(1, 4, key_vector_dim)
value = torch.randn(1, 4, value_vector_dim)
