
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.div(0.1)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.05)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model parameters
x1 = torch.randn(1, 128).requires_grad_()
x2 = torch.randn(1, 128, 256).requires_grad_()

# Compute forward pass
