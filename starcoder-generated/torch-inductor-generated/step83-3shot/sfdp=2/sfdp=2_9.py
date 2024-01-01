
class Model(torch.nn.Module):
    def __init__(self, dim, inverse_scale_factor, dropout_rate):
        super().__init__()
        self.dropout_p = dropout_rate

    def forward(self, query, key, value, input_mask):
       qk = torch.matmul(query, key.transpose(-2, -1))
       inv_scale_factor = torch.tensor(self.d_model**0.5, device=qk.device)
       scaled_qk = qk.div(inv_scale_factor)
       dropout_qk = torch.nn.functional.dropout(scaled_qk, p=self.dropout_p)
       output = dropout_qk.matmul(value) + input_mask

# Initializing the model
m = Model(dim=dim, inverse_scale_factor=inverse_scale_factor, dropout_rate=dropout_rate)

# Inputs to the model
query = torch.randn(3, 5, dim)
key = torch.randn(3, 8, dim)
value = torch.randn(3, 8, dim)
input_mask = torch.randn(3, 5, 8)
