
class Model(torch.nn.Module):
    def forward(self, query, key, value, input_mask, dropout_p):
        assert query.dim() == 2, 'The rank of query must be 2.'
        assert key.dim() == 2, 'The rank of key must be 2.'
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1), 'The sizes of the key and the value must be equal.'
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor) 
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) 
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 6) 
key = torch.randn(2, 6) 
value = torch.randn(2, 8)
dropout_p = 0.0
input_mask = None
