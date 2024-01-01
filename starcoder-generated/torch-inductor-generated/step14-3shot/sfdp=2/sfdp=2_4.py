
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(2, 8, 20)
key = torch.randn(2, 8, 20)
value = torch.randn(2, 8, 20)
inv_scale_factor = torch.randint(0,63, (2,)).float().abs().div(128)
dropout_p = torch.randint(0,63, (2,)).float().abs().div(128)
