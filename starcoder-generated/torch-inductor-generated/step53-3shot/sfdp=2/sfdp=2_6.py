
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p):
        super().__init__()
        self.input_dim = input_dim
 
    def _make_linear(self, input_dim, output_dim, bias, dropout_p):
        return [ torch.nn.Linear(input_dim, output_dim, bias=bias),
                         torch.nn.Dropout(dropout_p) ]
 
    def _make_attention(self, feature_dim, dropout_p):
        return [ torch.nn.Linear(feature_dim, feature_dim),
                         torch.nn.Tanh(),
                         torch.nn.Linear(feature_dim, 1, bias=False),
                         torch.nn.Dropout(dropout_p) ]
 
    def forward(self, query, key, value, scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1.0 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing model
m = Model(input_dim=6, output_dim=4, dropout_p=0.2)
 
# Inputs to the model
query = torch.randn(4, 10, 6)
key = torch.randn(5, 20, 6)
value = torch.randn(5, 20, 4)
scale_factor = torch.randn(1)
