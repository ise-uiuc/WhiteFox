
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
        self.qk_scaling_factor = 1 / math.sqrt(self.attention_dim) # attention_dim is the hidden size of the query, key, and value tensors
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk * self.qk_scaling_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 10, 10)
key = torch.randn(1, 1, 10, 10)
value = torch.randn(1, 1, 10, 10)
