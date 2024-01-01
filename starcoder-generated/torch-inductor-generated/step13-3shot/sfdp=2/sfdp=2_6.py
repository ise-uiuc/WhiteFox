
class Model(torch.nn.Module):
    def __init__(self, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.query_proj = torch.nn.Linear(64, 64)
        self.key_proj = torch.nn.Linear(64, 64)
        self.value_proj = torch.nn.Linear(64, 64)
        self.inv_scale_factor = 1/(64**0.5)
 
    def forward(self, x1, dropout_p=0.1):
        q1 = self.query_proj(x1)
        k1 = self.key_proj(x1)
        v1 = self.value_proj(x1)
 
        q1 = self._reshape(q1)
        k1 = self._reshape(k1)
        v1 = self._reshape(v1)
 
        q2 = q1.transpose(-2, -1)
        k2 = k1
        v2 = v1
 
        q3 = torch.matmul(q2, k2)
        scaled_qk = q3.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v2)
 
        return self._reshape(output)
    
    def _reshape(self, x):
        new_x = x.reshape(*(-1, self.n_heads, x.size(-2), x.size(-1))).permute(0, 2, 1, 3)
        return new_x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
