
class Model(torch.nn.Module):
    def forward(self, q, k, v, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
def init_inputs():
    device = torch.device('cpu')
    q = torch.randn(12, 512, 64, 64, device=device)
    k = torch.randn(12, 512, 64, 64, device=device)
    v = torch.randn(12, 512, 64, 64, device=device)
    inverser_scale_factor = torch.ones([], device=device)
    return q, k, v, inverser_scale_factor

# Testing the model with the input
q, k, v, inverser_scale_factor = init_inputs()
