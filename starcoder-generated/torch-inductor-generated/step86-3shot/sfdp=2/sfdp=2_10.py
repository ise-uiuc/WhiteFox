
class Model(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_key, d_value):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_key = d_key
        self.d_value = d_value
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_key * num_heads)
        self.value = torch.nn.Linear(d_model, d_value * num_heads)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, x1):
        v1 = self.query(x1)
        v2 = self.key(x1)
        v3 = self.value(x1)
        q = v1.reshape(-1, self.num_heads, self.d_key)
        k = v2.reshape(-1, self.num_heads, self.d_key).transpose(-2, -1)
        v = v3.reshape(-1, self.num_heads, self.d_value)
        qk = torch.matmul(q, k)
        inv_scale_factor = self.d_key ** -0.5
        scaled_qk = qk * inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output.transpose(1, 2).reshape(-1, self.d_model)

# Initializing the model
m = Model(num_layers=2, d_model=256, num_heads=4, d_key=64, d_value=64)

# Inputs to the model
x1 = torch.randn(2, 256)
