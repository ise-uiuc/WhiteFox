
class Model(torch.nn.Module):
    def __init__(self, input3_dim=4, input4_dim=4, head_num=4, dropout_p=0.5):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.scale_factor = math.sqrt(input3_dim)
        torch.nn.init.normal_(self.scale_factor, mean=1.0, std=7.0)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.output = torch.nn.Linear(head_num, input4_dim)
 
    def forward(self, input1, input2, input3, input4):
        x1 = torch.matmul(input1, input2.transpose(-2, -1))
        x2 = x1 / self.scale_factor
        x3 = self.softmax(x2)
        x3 = self.dropout(x3)
        y1 = torch.matmul(x3, input4)
        output = self.output(y1)
        return output

# Initializing the model
m = Model().to(device)

# Inputs to the model
input1 = torch.randn(2, 3, 4).to(device)
input2 = torch.randn(2, 4, 3).to(device)
input3 = torch.randn(3, 3).to(device)
input4 = torch.randn(3, 4).to(device)
