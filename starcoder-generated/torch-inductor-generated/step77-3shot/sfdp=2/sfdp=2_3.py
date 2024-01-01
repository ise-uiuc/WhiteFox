
class ModelWithSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = (64 / 256)**0.25
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = self.scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.8)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = ModelWithSoftmax()

# Inputs to the model
query = torch.randn(4, 3, 64, 64)
key = torch.randn(4, 3, 64, 64)
value = torch.randn(4, 3, 64, 64)
