
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = math.sqrt(dimension_per_head)
        scaled = qk.div(inv_scale_factor)
        softmax = scaled.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 12, 12, 36)
key = torch.randn(1, 6, 12, 36)
value = torch.randn(1, 6, 12, 36)

