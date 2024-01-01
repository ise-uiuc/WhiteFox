
class Model(torch.nn.Module):
    def __init__(self, nb_features):
        super().__init__()
        self.nb_features = nb_features
        self.scale_factor = self.nb_features ** -0.5
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(x2)

# Inputs to the model
x1 = torch.randn(1, 10, 8, 8)
x2 = torch.randn(1, 10, 8, 8)
