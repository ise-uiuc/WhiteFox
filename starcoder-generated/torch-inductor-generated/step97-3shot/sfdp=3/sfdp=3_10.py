
class Model(nn.Module):
    def __init__(self, query_length, key_length, head_num, dropout_p=0.5):
        super().__init__()
        self.mat_mul = torch.nn.Linear(query_length, key_length)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.output = torch.nn.Linear(key_length, query_length)
 
    def forward(self, query, key, value):
        mat_mul = self.mat_mul(query)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(0.125)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = self.output(dropout_qk)
        return output
 
# Initializing the model
m = Model(10, 10, 10)
 
# Initializing the query, key, and value tensors
query = torch.randn(1, 10)
key = torch.randn(1, 10)
value = torch.randn(1, 10)
