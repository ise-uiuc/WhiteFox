
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q1, k):
        qk = q1 @ k.transpose(-2, -1) / math.sqrt(q1.size(-1))
        a = torch.nn.functional.one_hot(
            torch.arange(0, qk.shape[-2]).unsqueeze(0).repeat(qk.shape[0], 1),
            qk.shape[-2],
        ).type_as(qk)
        qk = qk + a
        aat = torch.softmax(qk, dim=-1).clone()
        output = aat @ k
        return output
# Inputs to the model
query5 = torch.randn(1, 10, 64)
key3 = torch.randn(1, 16, 64)
