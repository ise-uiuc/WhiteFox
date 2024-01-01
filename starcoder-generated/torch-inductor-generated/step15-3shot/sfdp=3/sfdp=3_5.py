
class Model(torch.nn.Module):
    def __init__(self, n_head, d_value, d_key, d_model, dim_feedforward, dropout_p):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, n_head, dropout=dropout_p)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
 
    def forward(self, x1):
        attn_output, _ = self.self_attn(x1, x1, x1)
        v1 = torch.nn.functional.relu(self.linear1(attn_output))
        v2 = self.dropout(v1)
        out = self.linear2(v2)
        return out

# Initializing the model
m = Model(n_head=2, d_value=64, d_key=64, d_model=32, \
          dim_feedforward=64, dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(35, 32, 64)
