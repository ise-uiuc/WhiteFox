
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = query @ key.transpose(-2, -1)
        scaled_qk = qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, 16, 16)
key = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
inv_scale_factor = torch.sqrt(32.0)
