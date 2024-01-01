s
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, scale_factor=None, inv_scale=None, dropout_p=0.0):
        qk = torch.matmul(q, k.transpose(-2, -1))
        if scale_factor is not None:  # The condition can be simplified via torch.matmul
            qk = qk.div(scale_factor.unsqueeze(-1)).div(inv_scale.unsqueeze(-1).unsqueeze(-2))
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(v)
 
# Initializing the models
model1 = Model1()
model2 = Model2()

# Inputs to the model
__input_q__ = torch.randn(5, 20, 8)
__input_k__ = torch.randn(5, 8, 100)
__input_v__ = torch.randn(5, 50, 8)

output1 = model1(__input_q__, __input_k__, __input_v__)
output2 = model2(__input_q__, __input_k__, __input_v__)

