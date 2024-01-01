
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(20, 20, bias=True)
        self.k = torch.nn.Linear(20, 20, bias=True)
 
    def compute_attention(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x2)
        output = self.compute_attention(query=q, key=k, value=v, scale_factor=f, dropout_p=p)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20, 10)
x2 = torch.randn(1, 20, 5)
