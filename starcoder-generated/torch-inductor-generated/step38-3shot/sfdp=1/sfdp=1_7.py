
class Model(torch.nn.Module):
    def __init__(self, dim=128, n_heads=8, dropout_p=0.01, inv_scale_factor=1):
        super().__init__()
        self.head_dim = dim//n_heads
        self.scale = self.head_dim**-0.5
        self.fc_0 = torch.nn.Linear(dim, dim)
        self.fc_1 = torch.nn.Linear(dim, dim, bias=False)
        self.fc_2 = torch.nn.Linear(dim, dim)
        self.fc_3 = torch.nn.Linear(dim, dim, bias=True)
        
        # torch.nn.init.ones_(self.fc_3.bias) # Initialize the bias of the last linear layer to 1
        # self.fc_3.bias.requires_grad = False # Prevent the bias of the last linear layer to learn
        self.fc_3.bias = torch.nn.Parameter(torch.zeros(dim), requires_grad=False) # Prevent the bias of the last linear layer to learn

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 16, 128)
key = torch.randn(2, 16, 128)
value = torch.randn(2, 16, 128)
