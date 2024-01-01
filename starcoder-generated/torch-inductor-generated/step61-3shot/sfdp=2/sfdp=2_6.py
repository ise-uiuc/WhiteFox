
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 64**0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = torch.matmul(dropout_qk, value)
        return output

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project = Linear(512, 1, bias=False)
 
    def forward(self, query, key, value):
        r1 = self.project(query, key, value)
        return r1

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 512, 4)
key = torch.randn(1,4,5,5)
value = torch.randn(1, 512, 4, 17, 17)

r1 = m(query, key, value)
