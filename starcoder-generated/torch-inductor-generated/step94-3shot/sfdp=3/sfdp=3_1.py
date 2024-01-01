
class Model(torch.nn.Module):
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
batch_size = 8
embed_dim = 128   
query = torch.randn(batch_size, seq_length, embed_dim)
key = torch.randn(batch_size, seq_length, embed_dim)
value = torch.randn(batch_size, seq_length, embed_dim)
scale_factor = 4.0
dropout_p = 0.2
m = Model()
