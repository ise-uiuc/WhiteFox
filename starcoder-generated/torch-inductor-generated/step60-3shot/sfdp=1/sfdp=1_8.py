
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2):
        qk: torch.Tensor = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk: torch.Tensor = qk.div(inv_scale_factor)
        softmax_qk: torch.Tensor = scaled_qk.softmax(dim=-1)
        dropout_qk: torch.Tensor = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output: torch.Tensor = dropout_qk.matmul(value)
        return output

# Initialize the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 2, 8)
x2 = torch.randn(4, 8, 2)
