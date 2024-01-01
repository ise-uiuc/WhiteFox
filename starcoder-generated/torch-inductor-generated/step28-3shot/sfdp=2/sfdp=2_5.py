
class Model(torch.nn.Module):
    def __init__(self, n_queries, n_keys, n_values, n_heads):
        super().__init__()
        self.q = torch.nn.Linear(n_queries, n_heads * n_queries)
        self.k = torch.nn.Linear(n_keys, n_heads * n_keys)
        self.v = torch.nn.Linear(n_values, n_heads * n_values)
 
    def forward(self, q_input, k_input, v_input):
        q = self.q(q_input)
        k = self.k(k_input)
        v = self.v(v_input)

#         q: (batch_size, n_q, n_heads * n_q)
#         k: (batch_size, n_k, n_heads * n_k)
#         v: (batch_size, n_k, n_heads * n_v)
 
        q, k, v = q.view(q.shape[0], -1, n_heads, -1), k.view(k.shape[0], -1, n_heads, -1), v.view(v.shape[0], -1, n_heads, -1)

#         q -> (batch_size, n_q, n_heads, n_q)
#         k -> (batch_size, n_k, n_heads, n_k)
#         v -> (batch_size, n_k, n_heads, n_v)
 
        q, k, v = q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3)

#         q: (batch_size, n_heads, n_q, n_q)
#         k: (batch_size, n_heads, n_k, n_k)
#         v: (batch_size, n_heads, n_v, n_v)
 
        qk = torch.matmul(q, k.transpose(-2, -1))

#         qk: (batch_size, n_heads, n_q, n_k)
 
        inv_scale_factor = math.sqrt(1. / math.sqrt(q_input.shape[-1]) + 1e-6)
#         inv_scale_factor = 0.5
 
#         inv_scale_factor = 0.2

        qk = qk.div(inv_scale_factor)
 
        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
 
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
n_queries = 10
n_keys = 10
n_values = 20
n_heads = 3
m = Model(n_queries, n_keys, n_values, n_heads)

# Inputs to the model
q_input = torch.randn(5, 6, n_queries)
k_input = torch.randn(6, 5, n_keys)
v_input = torch.randn(6, 5, n_values)
