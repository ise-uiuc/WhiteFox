
class Model(torch.nn.Module):
    def __init__(self, scale_factor, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale_factor = torch.nn.Parameter(torch.tensor(scale_factor))
    
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = softmax_qk.matmul(v)
        return torch.nn.functional.dropout(output, p=self.dropout_p)
        
# Initializing the model
m = Model(scale_factor=2, dropout_p=0.5)

# Inputs to the model
q, k, v = torch.randn(1, 6, 120, 128), torch.randn(1, 6, 120, 128), torch.randn(1, 6, 120, 128)
