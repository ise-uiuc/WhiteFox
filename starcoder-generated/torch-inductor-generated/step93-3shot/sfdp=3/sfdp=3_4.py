
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
        self.lin1 = torch.nn.Linear(in_features=5, out_features=10)
    
    def forward(self, q, k, v, scale_factor=0.1):
        q = self.lin1(q)
        k = self.lin1(k)
        v = self.lin1(v)
  
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk * scale_factor
        output = self.dropout(torch.nn.functional.softmax(scaled_qk, dim=-1)).matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(5, in_features=5)
k = torch.randn(5, in_features=5)
v = torch.randn(5, in_features=10)
scale_factor = 0.1
