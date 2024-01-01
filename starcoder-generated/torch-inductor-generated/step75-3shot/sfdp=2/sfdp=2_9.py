
class Model(torch.nn.Module):
    def __init__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout_p: float = 0.1):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value
        self.dropout_p = dropout_p
 
    def forward(self, x):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        inv_scale_factor = qk.size()[-1] ** -0.5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
query = torch.randn(8 * 8, 64)
key = torch.randn(8 * 8, 64)
value = torch.randn(8 * 8, 64)
m = Model(query, key, value)

# Inputs to the model
x = torch.randn(8, 8, 64)
