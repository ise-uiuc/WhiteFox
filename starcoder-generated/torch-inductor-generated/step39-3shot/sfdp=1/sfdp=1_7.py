
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(seq_length, batch_size, hidden_size)
key = torch.randn(seq_length, batch_size, hidden_size)
value = torch.randn(seq_length, batch_size, hidden_size)
