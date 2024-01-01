
class Model(torch.nn.Module):
    def __init__( self ):
        super().__init__()

    def forward( self, q, k, v, scale_factor, is_training=True ):
        q = torch.matmul(q, k.transpose(1, 0))
        scaled_qk = q.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5, training=is_training)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(3, 16)
k = torch.randn(16, 28)
v = torch.randn(16, 28)
scale_factor = 10.0
