
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.tensor([np.power(k.shape[-1], -0.5)]).to(device)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = torch.tensor([0.5]).to(device)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        v = torch.randn([q.shape[0], v.shape[1], q.shape[2]]).to(device)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(batch_count, dim_count)
k = torch.randn(batch_count, dim_count)
