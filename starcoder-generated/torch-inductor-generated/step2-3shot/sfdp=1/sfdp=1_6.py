
class Model(torch.nn.Module):
    def __init__(self, dim_q=1, dim_k=2, dim_v=3, scale=1, dropout_p=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.linear_q = nn.Linear(dim_q, dim_k)
        self.linear_k = nn.Linear(dim_k, dim_k)
        self.linear_v = nn.Linear(dim_v, dim_v)
        self.scale = scale
 
    def forward(self, x1, x2):
        query = self.linear_q(x1)
        key = self.linear_k(x2)
        value = self.linear_v(x2)
        qk = torch._empty_affine_quantized([query.size(0), query.size(1), key.size(1)], scale=self.scale, zero_point=0, dtype=torch.qint8)
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor(1.0/self.scale, device=qk.device)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(dim_q=5, dim_k=7, dim_v=5, scale=10)

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 7)
