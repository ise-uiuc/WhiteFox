
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, inv_scale_factor, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model(0.5)

# Inputs to the model
query = torch.randn(3, 512, 128)
key = torch.randn(3, 512, 128)
value = torch.randn(3, 512, 128)
inv_scale_factor = torch.randn(3)
