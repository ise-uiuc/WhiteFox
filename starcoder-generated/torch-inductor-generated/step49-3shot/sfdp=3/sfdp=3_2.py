
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        scale_factor = torch.tensor(kwargs.pop("scale_factor", 1.0))
        dropout_p = kwargs.pop("dropout_p", 0.0)
        self.dropout_p = dropout_p
        self.scale_factor = scale_factor
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scale_factor = self.scale_factor.to(xk.device) if self.scale_factor.is_floating_point() else self.scale_factor
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk * k

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 4, 5)
x2 = torch.rand(2, 4, 5)
