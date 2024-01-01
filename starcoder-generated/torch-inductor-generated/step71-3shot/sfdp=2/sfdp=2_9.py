
class QueryKey(torch.nn.Module):
    def forward(self, query, key):
        qk = torch.matmul(query, key.transpose(-2, -1))
        return qk

class ScaledDotProduct(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_scale_factor = 1. / math.sqrt(8.)

    def forward(self, query, key):
        qk = self.qk(query, key)
        scaled_qk = qk.div(self.inv_scale_factor)
        return torch.softmax(scaled_qk, dim=-1)

class DropoutApply(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.6

    def forward(self, query, key):
        softmaxed = self.sdp(query, key)
        dropout = torch.nn.functional.dropout(softmaxed, p=self.dropout_p)
        return dropout 

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = DropoutApply()
        self.matmul = torch.matmul 

    def forward(self, query, key, value):
        return self.matmul(self.dropout(query, key), value)

# Initializing the model
qk = QueryKey()
sdp = ScaledDotProduct()
dropout = DropoutApply()
matmul = torch.matmul 
att = Attention()

# Inputs to the model
query = torch.randn(1, 4, 64)
key = torch.randn(1, 4, 64)
value = torch.randn(1, 4, 64)
