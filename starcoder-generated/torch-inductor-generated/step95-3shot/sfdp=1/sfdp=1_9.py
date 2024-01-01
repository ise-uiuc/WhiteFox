
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(80, 100)
        self.key = torch.nn.Linear(80, 100)
        self.value = torch.nn.Linear(80, 100)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key):
        q = self.query(query)
        k = self.key(key)
        v = self.value(key)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(10)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.05)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 80)
x2 = torch.randn(1, 80)
