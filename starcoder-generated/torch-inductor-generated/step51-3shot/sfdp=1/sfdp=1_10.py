
class Model(torch.nn.Module):
    def __init__(self, scale_factor=1, dropout_p=0, device='cpu'):
        super().__init__()
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor(self.scale_factor).to(self.device).inverse()
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 2, 200)
key =  torch.randn(3, 2, 500)
value = torch.randn(3, 2, 500)
