
class Model(torch.nn.Module):
    def __init__(self, vocab_size, dim, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(vocab_size, dim, dropout_p)

# Inputs to the model
query = torch.randn(10, 32, embedding_dim)
key = torch.randn(10, 100, embedding_dim)
value = torch.randn(10, 100, embedding_dim)
scale_factor = 0.2
