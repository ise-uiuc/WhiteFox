
class Model(torch.nn.Module):
    def __init__(self, dropout_p, d_k, d_model, n_heads, scale_factor):
        super().__init__()
        self.dropout1 = torch.nn.Dropout(dropout_p)
        self.dropout2 = torch.nn.Dropout(dropout_p)
        self.d_k, self.d_model, self.n_heads = d_k, d_model, n_heads
        self.linear1 = torch.nn.Linear(d_model, d_k * n_heads)
        self.linear2 = torch.nn.Linear(d_model, d_k * n_heads)
        self.linear3 = torch.nn.Linear(d_k * n_heads, d_model)
        self.scale_factor = scale_factor
 
    def forward(self, q, k, v):
        x1 = self.linear1(q).view(q.size(0), q.size(1), self.n_heads, self.d_k)
        x2 = self.linear2(k).view(k.size(0), k.size(1), self.n_heads, self.d_k)
        x3 = self.linear3(v).view(v.size(0), v.size(1), self.n_heads, self.d_k)
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout1(softmax_qk)
        output = self.dropout2(
                torch.matmul(dropout_qk, x3).view(v.size(0), v.size(1), self.n_heads * self.d_k))
        return output

# Initializing the model
m = Model(dropout_p=0.5, d_k=64, d_model=256, n_heads=8, scale_factor=self.d_k ** 0.5)

# Inputs to the model
x1 = torch.randn(1, 2, 3, 256)
x2 = torch.randn(1, 2, 2, 256)
x3 = torch.randn(1, 2, 3, 256)
v1 = m(x1, x2, x3)

