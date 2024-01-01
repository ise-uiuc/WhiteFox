
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.scale_factor = 1 / math.sqrt(d_k)
        self.dropout_p = 0.1
        weight_list = []
        for i in range(n_head):
            for j in range(d_v):
                weight = torch.nn.Parameter(torch.randn(d_model, 1, d_k))
                weight_list.append(weight)
        self.weight_list = torch.nn.ParameterList(weight_list)
 
    def forward(self, q, k, v):
        # Reshape q, k, v and combine q, k, v into new `input` tensor
        new_q, new_k, new_v = None, None, None
        for i in range(self.n_head):
            # reshape q, k, v
            q_i_reshape = q[i:q.shape[0]:self.n_head]
            k_i_reshape = k[i:k.shape[0]:self.n_head]
            v_i_reshape = v[i:v.shape[0]:self.n_head]
            # transpose k, v and create new_k, new_v
            k_i_transpose = torch.transpose(k_i_reshape, 1, 2)
            new_k_i = torch.matmul(q_i_reshape, k_i_transpose) * self.scale_factor
            k_i_transpose = torch.transpose(k_i_reshape, 1, 2)
            new_v_i = torch.matmul(q_i_reshape, k_i_transpose)
            if i == 0:
              new_q = q_i_reshape
              new_k = new_k_i
              new_v = new_v_i
            else:
              new_q = torch.cat([new_q, q_i_reshape], 2)
              new_k = torch.cat([new_k, new_k_i], 2)
              new_v = torch.cat([new_v, new_v_i], 2)
        # new_q has shape [n_batch, seq_len, d_model]
        # new_k has shape [n_batch, seq_len, seq_len]
        # new_v has shape [n_batch, seq_len, seq_len]
        # Compute softmax and dropout
        softmax = torch.nn.Softmax(dim=2)
        dropout = torch.nn.Dropout(self.dropout_p)
        softmax_qk = softmax(new_k)
        dropout_qk = dropout(softmax_qk)
        # Compute the dot product of the dropout output and the value tensor
        output = torch.matmul(dropout_qk, new_v)
        return output

# Initializing the model
n_head = 2
d_model = 512
d_k = 64
d_v = 64
m = Model(n_head, d_model, d_k, d_v)

# Inputs to the model
n_batch = 1
seq_len = 8
q = torch.randn(n_batch, seq_len, d_model)
k = torch.randn(n_batch, seq_len, d_model)
v = torch.randn(n_batch, seq_len, d_model)
