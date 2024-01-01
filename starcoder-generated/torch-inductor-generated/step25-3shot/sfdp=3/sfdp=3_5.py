
class Model(torch.nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p=0.5):
        super().__init__()
        self.dim_model = dim_model
        self.dropout_p = dropout_p
        self.scale_factor = np.sqrt(self.dim_model // num_heads)

        self.linear_q = torch.nn.Linear(self.dim_model, self.dim_model)
        self.linear_k = torch.nn.Linear(self.dim_model, self.dim_model)
        self.linear_v = torch.nn.Linear(self.dim_model, self.dim_model)
 
    def forward(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
model = Model(256, 8)

# Inputs to the model
query = torch.randn(2, 64, 256)
key = torch.randn(2, 128, 256)
value = torch.randn(2, 128, 256)
