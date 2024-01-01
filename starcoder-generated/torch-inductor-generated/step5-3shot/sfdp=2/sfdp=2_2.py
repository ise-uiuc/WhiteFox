
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, n_hidden_layers, n_hidden_nodes, dropout_p, attention_dropout_p, inv_scale_factor):
        super().__init__()
        self.query_linear1 = torch.nn.Linear(query_size, n_hidden_nodes)
        self.value_linear1 = torch.nn.Linear(value_size, n_hidden_nodes)
        for i in range(n_hidden_layers):
            self.query_linear.append(torch.nn.Linear(n_hidden_nodes, n_hidden_nodes))
            self.value_linear.append(torch.nn.Linear(n_hidden_nodes, n_hidden_nodes))

    def forward(self, query, key, value):
        q = self.query_linear[0](query)
        v = self.value_linear[0](value)
        for i in range(1, n_hidden_layers):
            q = self.query_linear[i](q)
            v = self.value_linear[i](v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model(query_size, key_size, value_size, n_hidden_layers, n_hidden_nodes, dropout_p, attention_dropout_p, inv_scale_factor)

# Inputs to the model
query = torch.randn(1, n_queries, query_size)
key = torch.randn(1, n_keys, query_size)
value = torch.randn(1, n_keys, value_size)
