
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = query @ key.transpose(-2, -1)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk @ value
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(8, 64, 512)
key = torch.randn(4, 1, 512)
value = torch.randn(4, 64, 512)
inv_scale_factor = 1. / math.sqrt(math.sqrt(512))
dropout_p = 0.875
