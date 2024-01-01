
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = math.sqrt(128)
        self.dropout_p = 0.75

    def forward(self, __input__):
        kq = torch.matmul(__input__, __input__.transpose(-2, -1))
        scaled_kq = kq.mul(self.scale_factor)
        softmax_kq = scaled_kq.softmax(dim=-1)
        dropout_kq = torch.nn.functional.dropout(softmax_kq, p=self.dropout_p)
        output = dropout_kq.matmul(__input__)
        return output

# Initializing the model
m = Model()

# Inputs to the model
