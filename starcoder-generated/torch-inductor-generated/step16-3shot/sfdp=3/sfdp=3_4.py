
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = torch.nn.Linear(512, 512, bias=False)
        self.linear_k = torch.nn.Linear(512, 512, bias=False)
        self.linear_v = torch.nn.Linear(512, 512, bias=False)
        self.dropout = torch.nn.Dropout(0.5)
        self.scale_factor = 1.0 / math.sqrt(self.linear_q.out_features)
 
    def forward(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(16, 8, 512)
key = torch.randn(16, 8, 512)
value = torch.randn(16, 8, 512)
