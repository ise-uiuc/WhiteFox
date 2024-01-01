
class Model(torch.nn.Module):
    def __init__(self, n_head, d_model, d_query_key, d_value, dropout_p=0.1, inv_scale_factor=1.0):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_query_key = d_query_key
        self.d_value = d_value
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
        self.weight_query = torch.nn.Parameter(torch.FloatTensor(n_head, d_query_key, int(d_model / d_query_key)))
        # Weights for the key tensor are initialized in the same way as the weights for the query tensor.
        self.weight_value = torch.nn.Parameter(torch.FloatTensor(n_head, d_query_key, int(d_model / d_query_key)))
 
    def forward(self, query, key, value):
        q = torch.matmul(query, self.weight_query).view([query.size(0), query.size(1), self.n_head, self.d_query_key])
        k = torch.matmul(key, self.weight_query).view([key.size(0), key.size(1), self.n_head, self.d_query_key])
        # The key tensor is split into n_head heads of query keys.
        # The query tensor is split int n_head heads of query keys.
        # This allows the computation of the query key dot products by weighting the corresponding keys and adding the corresponding query key vectors.
        v = torch.matmul(value, self.weight_value).view([value.size(0), value.size(1), self.n_head, self.d_query_key])
        # The value tensor is split into n_head heads of query keys.
        # This allows the computation of the value vector dot products by weighting the corresponding values and adding the corresponding value vectors.
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        # Weight the dot product by the inverse scale factor
        scaled_qk = qk.div(self.inv_scale_factor)
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        # The dropout output is multiplied by the query keys and value vectors and then the sums are computed.
        # This is the typical calculation process in Transformer, the output dimension of this layer is n_head * dv = n_head * d_value.
        output = output.reshape([query.size(0), query.size(1), -1])
        # The output is transformed into n_head dim, this is because the sum of dv dimension is the same as the hidden dimension, and the last dimension dv * n_head = d_model, that is, the output dimension is the same as the input dimension hidden dimension.
        return output

# Initializing the model
m = Model(n_head, d_model, d_query_key, d_value)

# Inputs to the model
query = torch.randn(1, 16, hidden_dim) if batch_size is None else torch.randn(batch_size, 16, hidden_dim)
key = torch.randn(1, 40, hidden_dim) if batch_size is None else torch.randn(batch_size, 40, hidden_dim)
value = torch.randn(1, 40, hidden_dim) if batch_size is None else torch.randn(batch_size, 40, hidden_dim)
