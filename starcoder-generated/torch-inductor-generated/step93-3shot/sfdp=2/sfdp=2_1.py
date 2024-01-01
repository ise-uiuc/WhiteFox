
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def _forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
    def forward(self, query, key, value):
        return self._forward(query, key, value)
 
    def inference(self, query, key, value, inv_scale_factor=1.0, dropout_p=0.0):
        with torch.no_grad():
            return self._forward(query, key, value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 2)
key = <KEY>")
value = torch.Tensor([[[[ 1.2744, -0.4185],
                         [ 1.2433,  1.2546]],
   [[ 1.5079,  0.8890],
    [ 0.1855,  0.0166]],

   [[ 0.8599, -1.0796],
    [ 1.1591, -0.4120]]]])
