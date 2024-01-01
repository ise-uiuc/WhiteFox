
class Model(torch.nn.Module):
    def __init__(self, dropout_p = 0.1):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, scale_factor = None):
        if scale_factor == None:
            inv_scale_factor = None
        else:
            if isinstance(scale_factor, torch.Tensor):
                inv_scale_factor = scale_factor.reciprocal()
            else:
                inv_scale_factor = 1 / scale_factor
        qk = torch.matmul(query, key.transpose(-2, -1))
        if inv_scale_factor == None:
            scaled_qk = qk
        else:
            scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model(dropout_p = 0.1)

# Inputs to the model
query = torch.randn(1, 1, 128)
key = torch.randn(1, 3, 128)
value = torch.randn(1, 3, 128)
