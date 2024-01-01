
class Model(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.linear0 = torch.nn.Linear(in_channels, 128)
        self.linear1 = torch.nn.Linear(in_channels, 128)
        self.tanh = torch.nn.ReLU()
 
    def forward(self, x1, x2):
        lin0_out = self.linear0(x1)
        lin1_out = self.linear1(x2)
        qk = torch.matmul(lin0_out, lin1_out.transpose(-2, -1))
        scaled_qk = qk.div(16)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(lin0_out)
        return output

# Initializing the model
m = Model(16)

# Inputs to the model
x1 = torch.randn(2, 16)
x2 = torch.randn(2, 16)
