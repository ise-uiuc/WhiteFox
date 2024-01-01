
class Model(torch.nn.Module):
    def __init__(self, d_input, d_model, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.q_linear = torch.nn.Linear(d_input, d_model)
        self.k_linear = torch.nn.Linear(d_input, d_model)
        self.v_linear = torch.nn.Linear(d_input, d_model)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor):
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return dropout_qk.matmul(v)

# Initializing the model
m = Model(d_input=3, d_model=2, dropout_p=0.3)

# Inputs to the model
query = torch.randn(1, 3)
key = torch.randn(2, 6)
value = torch.randn(2, 6)
inv_scale_factor = torch.tensor(4.0)
