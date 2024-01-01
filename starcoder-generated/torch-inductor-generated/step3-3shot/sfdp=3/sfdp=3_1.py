
class Model(torch.nn.Module):
    def __init__(self, emb, heads, dropout_p):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.scale_factor = emb ** 0.5
        self.dropout_p = dropout_p
 
        self.k_conv = torch.nn.Conv2d(3, emb, 1, stride=1, padding=1)
        self.q_conv = torch.nn.Conv2d(3, emb, 1, stride=1, padding=1)
        self.v_conv = torch.nn.Conv2d(3, emb, 1, stride=1, padding=1)
        self.o_conv = torch.nn.Conv2d(emb, emb, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        k = self.k_conv(x1)
        q = self.q_conv(x2)
        v = self.v_conv(x1)
        scale_factor = self.scale_factor
        dropout_p = self.dropout_p

        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        