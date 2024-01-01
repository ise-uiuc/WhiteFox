
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, 1, 1, 1))
        self.key = torch.nn.Parameter(torch.randn(1, 1, 1, 1))
        self.value = torch.nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, inv_scale_factor):        
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
inv_scale_factor = torch.tensor(0.5) # Inverse scale factor to be used in the model
