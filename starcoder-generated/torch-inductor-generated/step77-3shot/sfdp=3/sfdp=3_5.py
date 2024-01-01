
class Model(torch.nn.Module):
     def __init__(self, m=384, n=256):
         super().__init__()
         self.lin1 = torch.nn.Linear(n, m)
         self.lin2 = torch.nn.Linear(m, n)
 
     def forward(self, x1, x2, scale_factor=1, dropout_p=0.0):
         w1 = self.lin1(x1)
         w2 = self.lin1(x2)
         v1 = torch.matmul(w1, w2.transpose(1,2))
         v2 = v1.mul(scale_factor)
         v3 = torch.nn.functional.softmax(v2, dim=-1)
         v4 = torch.nn.functional.dropout(v3, p=dropout_p)
         output = torch.matmul(v4, x1)
         return output

# Initializing the model
n = 512
m = 256
scale_factor = 10
dropout_p = 0.2
m = Model(m=m, n=n)

# Inputs to the model
x1 = torch.randn(1, n, requires_grad=True)
x2 = torch.randn(1, m, requires_grad=True)
