
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = torch.nn.Linear(200, 200)
        self.softmax_proj = torch.nn.Linear(200, 200)
        self.dropout_proj = torch.nn.Linear(200, 200)
        self.value_proj = torch.nn.Linear(300, 200)
        self.gelu = torch.nn.functional.gelu
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, q, k, v, inv_scale_factor):
        q = self.query_proj(q)
        k = self.query_proj(k)
        v = self.query_proj(v)
        q *= inv_scale_factor
        qk = torch.matmul(q, k.transpose(-2, -1))
        softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
        return output
 
    
# Initializing the model
m = Model()
 
# Inputs to the model
q = torch.randn(3, 200)
k = torch.randn(3, 200)
v = torch.randn(3, 200, 300)
inv_scale_factor = 10.0
