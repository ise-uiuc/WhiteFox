
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(torch.rand(256, 512))
        self.k = torch.nn.Parameter(torch.rand(256, 512))
        self.v = torch.nn.Parameter(torch.rand(256, 512))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout()
 
    def forward(self, query, value, key, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / 512
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 256, 512)
value = torch.randn(8, 256, 512)
key = torch.randn(8, 256, 512)
dropout_p = 0.1
