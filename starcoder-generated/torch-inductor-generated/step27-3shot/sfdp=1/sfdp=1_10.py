
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        self.query = torch.nn.Linear(128, 128, bias=False)
        self.key = torch.nn.Linear(128, 128, bias=False)
        self.value = torch.nn.Linear(128, 128, bias=False)
 
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value):
        inv_scale_factor = math.sqrt(128)
 
        q = self.query(query)
        k = self.key(key)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        
        p = 0.25
        dropout_soft = torch.nn.functional.dropout(softmax_qk, p=p)
 
        return self.value(dropout_soft.matmul(value))

# Initializing the model
m = Model2()

# Inputs to the model
query = torch.randn(5, 128)
key = torch.randn(5, 128)
value = torch.randn(5, 128)
