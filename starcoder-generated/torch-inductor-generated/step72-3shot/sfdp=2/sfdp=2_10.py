
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.qk_layer = torch.nn.Linear(query_dim, key_dim, bias=False)
        self.dropout_layer = torch.nn.Dropout(dropout_p)
        self.output_layer = torch.nn.Linear(key_dim, value_dim, bias=False)
    
    def forward(self, q, k, v):
        num_queries = q.shape[1]
        qk = self.qk_layer(q)
        k_t = k.transpose(2, 3)
        qk = torch.matmul(qk, k_t) # Matrix multiplication
        inv_scale_factor = 1 / np.sqrt(self.output_layer.in_features)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout_layer(softmax_qk)
        output = self.output_layer(dropout_qk)
        output = output.reshape(-1, num_queries, output.shape[-2], output.shape[-1])
        return output

# Initializing the model
input_shape = (1, 32, 128)
query_dim, key_dim, value_dim = 32, 64, 32
dropout_p = 0.5
m = Model(query_dim, key_dim, value_dim, dropout_p)

# Inputs to the model
q, k, v = torch.randn(input_shape), torch.randn(input_shape), torch.randn(input_shape)
