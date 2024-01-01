
class Model(torch.nn.Module):
    def __init__(self, tgt_len, src_len, bsz):
        super().__init__()
        self.linear = torch.nn.Linear(src_len, tgt_len)
        self.src_len = src_len
        self.tgt_len = tgt_len
        self.bsz = bsz

    def forward(self, q, k, v, mask):
        