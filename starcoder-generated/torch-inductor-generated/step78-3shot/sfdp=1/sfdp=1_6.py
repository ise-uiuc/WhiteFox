
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float, device="cpu"))
        self.key = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float, device="cpu"))
        self.value = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float, device="cpu"))
        self.inv_scale_factor = torch.nn.Parameter(torch.tensor(0.1767767, dtype=torch.float, device="cpu"))
        self.dropout_p = torch.nn.Parameter(torch.tensor(0.3568764, dtype=torch.float, device="cpu"))
 
    def forward(self, x3):
        qk = x3.matmul(self.query).matmul(self.key.transpose(-2, -1))
        sc_qk = qk.div(self.inv_scale_factor)
        smx_qk = sc_qk.softmax(dim=-1)
        dr_qk = dropout(smx_qk)
        x6 = dr_qk.matmul(self.value)
        return x6

x3 = torch.randn(4, 2, 2)
