
class Model(torch.nn.Module):
    def __init__(self, query, key, value, inv_scale_factor, dropout_p):
        super().__init__()
        self.scale1 = torch.nn.Parameter(torch.ones(query.shape[-1:]))
        self.dropout = torch.nn.functional.dropout(query, p=dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk / self.scale1
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = softmax_qk * inv_scale_factor.unsqueeze(-1) # Compute dropout_qk here
        return torch.matmul(dropout_qk, value)
 
# Initializing the model
m = Model(query, key, value, inv_scale_factor, dropout_p)

# Inputs to the model
query = torch.randn(1, 8, 128)
