
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn((8,8,96,96)))
        self.key = torch.nn.Parameter(torch.randn((8,8,224,224)))
        self.value = torch.nn.Parameter(torch.randn((8,8,224,224)))

    def forward(self):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()
<fim_middle>
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def foward(self, X, Y, Z):
        v1 = torch.einsum('bchw,bcij->baij', (X, self.kernel))
        v1 = torch.matmul(v1, self.kernel)
        v1 = torch.einsum('bchw,bkli->blhi', (X, self.kernel))
