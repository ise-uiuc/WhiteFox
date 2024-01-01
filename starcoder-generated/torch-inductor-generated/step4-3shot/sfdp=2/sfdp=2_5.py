
class MyModel(nn.Module):
    def __init__(self, query, key, value):
        super().__init__()
         
        self.scale_factor = math.sqrt(key.size(-1))
        self.dropout_p = 0.1
         
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.matmul = torch.matmul
 
    def forward(self, query):
        qk = self.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout = self.dropout(self.softmax_qk)
        ouput = self.matmul(dropout, value)
        return output

# Initializing the model
query = torch.randn(3, 2, 5)
key = torch.randn(3, 5, 6)
value = torch.randn(3, 6, 7)
m = MyModel(query, key, value)

# Inputs to the model
x1 = torch.rand(3, 2, 5)
