
class Model(torch.nn.Module):
    def __init__(self, d_model, scale_factor, dropout_p):
        super().__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model(d_model=512, scale_factor=10.0, dropout_p=0.1)
 
# Inputs to the model
query = torch.randn(500, 1024, 512)
key = torch.randn(500, 1024, 512)
value = torch.randn(500, 1024, 512)
