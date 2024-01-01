
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        inv_scalar_factor = 1.0 / np.sqrt(d_model)
        self.softmax_dropout = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Softmax(dim=-1)
        )
        self.mat_mul = torch.nn.Linear(d_model, d_model)
 
    def forward(self, query, key, value, mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax_dropout(scaled_qk)
        dropoput_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        attn_out = dropoput_qk.matmul(value)
        return attn_out


# Initializing the model
d_model = 512
num_heads = 8
m = Model(d_model, num_heads)

# Inputs to the model
query = torch.randn(8, 64, 512)
key = torch.randn(8, 64, 512)
value = torch.randn(8, 64, 512)
mask = torch.ones(8, 64, 64)
