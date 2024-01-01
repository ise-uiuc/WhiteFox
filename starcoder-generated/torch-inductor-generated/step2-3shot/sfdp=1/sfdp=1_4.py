
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.9912230090574967
        self.query = torch.reshape(torch.tensor([0.5083694933224469, 0.6863057290813249, 0.8547938882381287, 0.8540438219279745, 0.7649201749868487], dtype=torch.float32), (1, 5, 1))
        self.key = torch.reshape(torch.tensor([0.6470188223491024, 0.5337367124645433, 0.744625303168531, 0.20979824216253413, 0.9836990738279748, 0.5026671238557882, 0.9765962122000682, 0.7203937932716684, 0.8190525255649841, 0.7890955963764558, 0.9055223942802718], dtype=torch.float32), (1, 3, 2))
        self.value = torch.reshape(torch.tensor([0.43320631732334563, 0.7089893374715701, 0.7080458488440063, 0.47934023774931226, 0.8005637200642821, 0.7092149744937534, 0.7039069089135204, 0.7827379388400436, 0.47628791660427877], dtype=torch.float32), (1, 3, 2))
        self.scale_factor = 0.9281684039410877
 
    def forward(self, x1):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scale_factor = torch.nn.functional.relu(self.scale_factor)
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        o6 = torch.nn.functional.softmax(dropout_qk, dim=-1)
        o7 = torch.matmul(o6, self.value)
        return o7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 5)
