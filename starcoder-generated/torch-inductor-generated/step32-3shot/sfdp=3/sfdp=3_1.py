
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, q, k, v, scale_factor=None):
        qk = torch.matmul(q, k.transpose(-2, -1))
        if scale_factor is not None:  # If scale factor is not none, scale the output of the dot product
            scaled_qk = qk.mul(scale_factor)
        else:
            scaled_qk = qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        return dropout_qk.matmul(v)

#Initializes the model
m = Model()

# Inputs to the model
q = torch.randn(1, 16, 64, 64)
k = torch.randn(1, 16, 64, 64)
v = torch.randn(1, 16, 64, 64)
