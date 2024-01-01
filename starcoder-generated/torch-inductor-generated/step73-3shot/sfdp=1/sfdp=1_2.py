
class Model(torch.nn.Module):
    def __init__(self, query_size = 64, value_size = 64):
        super().__init__()
        self.q_proj = torch.nn.Linear(query_size, 32)
        self.k_proj = torch.nn.Linear(32, 8)
        self.v_proj = torch.nn.Linear(32, 8)
        self.scale_factor = 8
        self.dropout_p = 0.9935
        
    def forward(self, x1):
        q = self.q_proj(x1)
        k = self.k_proj(q)
        v = self.v_proj(q)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64)
