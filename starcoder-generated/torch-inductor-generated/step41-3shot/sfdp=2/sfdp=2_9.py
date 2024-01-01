
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Linear(n_features_key, h_size)
        self.query = torch.nn.Linear(n_features_query, h_size)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
key = torch.randn(1, n_queries, n_features_key)
query = torch.randn(1, n_keys, n_features_query)
value = torch.randn(1, n_keys, n_features_value)
inv_scale_factor = 0.5
dropout_p = 0.5
