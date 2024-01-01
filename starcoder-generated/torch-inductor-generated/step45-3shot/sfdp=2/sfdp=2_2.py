
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.nn.Linear(16, 16)
 
    def forward(self, x1, x2):
        qk = self.matmul(x1)
        inv_scale_factor = x2.norm(2, 1).clamp(min=1e-12).reciprocal()  
        softmax_qk = (qk.div(inv_scale_factor)).softmax(dim=-1)    
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(8, 16)
