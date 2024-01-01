
class Model(torch.nn.Module):
    def __init__(self, n_q, n_k, d_qk, d_v, dropout_p=0.0):
        super().__init__()
        self.n_q = n_q
        self.n_k = n_k
        self.d_qk = d_qk
        self.d_v = d_v
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale=1.0):
        v_query = query.view(-1, self.n_q * self.d_qk)
        v_key = key.view(-1, self.n_k * self.d_qk)
        v_value = value.view(-1, self.n_k * self.d_v)
        v_qk = torch.matmul(v_query, v_key.transpose(-2, -1))
        v_scaled_qk = v_qk.div(scale)
        v_softmax_qk = torch.nn.functional.softmax(v_scaled_qk, dim=-1)
        v_dropout_qk = torch.nn.functional.dropout(v_softmax_qk, p=self.dropout_p)
        v_output = torch.matmul(v_dropout_qk, v_value)
        v_output = v_output.view(query.shape[:-1] + (-1,))
        return v_output

# Initializing the model
m = Model(n_q=16, n_k=16, d_qk=8, d_v=8, dropout_p=0.0)

# Inputs to the model
q = torch.randn(3, 80, 8)
k = torch.randn(3, 160, 4)
v = torch.randn(3, 160, 8)
