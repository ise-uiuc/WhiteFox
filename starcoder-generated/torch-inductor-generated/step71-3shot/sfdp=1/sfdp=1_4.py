
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inverse_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inverse_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 128, 4) # [batch_size, sequence_length, embedding_size]
key = torch.randn(16, 128, 4) # [batch_size, sequence_length, embedding_size]
value = torch.randn(16, 128, 4) # [batch_size, sequence_length, embedding_size]
inverse_scale_factor = torch.tensor(1.0) # [1]
dropout_p = torch.tensor(0.0) # [1]
