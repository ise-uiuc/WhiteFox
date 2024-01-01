
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 32)
 
    def forward(self, x1, x2):
        k = self.embed(x1)
        q = self.embed(x2)
        v = self.embed(x1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = np.sqrt(k.shape[-1])
        softmax_qk = qk.div(inv_scale_factor).softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

x1 = torch.randint(3, (4, 8)).long()
x2 = torch.randint(3, (4, 4)).long()
