
class Model(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads=4,
        dropout_p=0.5,
        head_dim=None,
        scale_factor=math.sqrt(0.5)
    ):
        super().__init__()
        
        if head_dim is None:
            head_dim = input_dim // num_heads # The dimension for each attention head. 
        
        self.qk = torch.nn.Linear(input_dim, input_dim)
        self.value = torch.nn.Linear(input_dim, input_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.scale_factor = scale_factor

    def forward(self, x):
        qk = self.qk(x)
        v = self.value(x)
        attn = torch.matmul(qk, v.transpose(-2, -1))
        scale_attn = attn.mul(self.scale_factor)
        softmax_attn = scale_attn.softmax(dim=-1)
        dropout_attn = self.dropout(softmax_attn)
        output = torch.matmul(dropout_attn, v)
        return output

# Initializing the model
m = Model(input_dim=64)

# Inputs to the model
x = torch.randn(1, 64, 20)
