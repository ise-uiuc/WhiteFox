
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads, scaling=True, dropout=0.1):
        super().__init__()
        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value, inv_scale_factor):
        q = self.q_linear(query)
        k = self.k_linear(key)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing a simple BERT model with one attention layer
m = Model(d_model=16, num_heads=2)

# Inputs to the model
query = torch.randn(1, 2, 16)
key = torch.randn(1, 2, 16)
value = torch.randn(1, 2, 16)
inv_scale_factor = torch.tensor(1.)
dropout_p = 0.1

