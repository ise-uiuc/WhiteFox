
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_head_size = None
        self.hidden_size = None
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1)).div(self.attention_head_size ** 0.5)
        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()
m.attention_head_size = 64
m.hidden_size = 384

# Inputs to the model
query = torch.randn(10, 5, 64)
key = torch.randn(10, 3, 64, 64)
value = torch.randn(10, 4, 64, 32)
dropout_p = 0.2
