
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 0.125
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.drop(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 128, 512)
key = torch.randn(1, 128, 512)
value = torch.randn(1, 128, 512)
