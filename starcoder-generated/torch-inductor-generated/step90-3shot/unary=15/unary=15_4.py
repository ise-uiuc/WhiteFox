
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.reshape([1, 3, 5, 5])
        v2 = v1.transpose([0, 3, 2, 1])
        v3 = v2.reshape([1, 6, 5, 5])
        v4 = v3.transpose([0, 3, 2, 1])
        v5 = v4.reshape([1, 10, 5, 5])
        v6 = torch.nn.functional.embedding(v5, torch.eye(3), 13, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
        return v6
# Inputs to the model
x1 = torch.randint(-2, 3, [1, 3, 5, 5])
