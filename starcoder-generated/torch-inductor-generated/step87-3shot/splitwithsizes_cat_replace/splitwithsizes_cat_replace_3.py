
model = nn.Sequential(OrderedDict([
    ('conv2d', nn.Conv2d(3, 32, (3, 36), stride=(2, 1), padding=(3, 18))),
    ('batch_norm2d', nn.BatchNorm2d(32)),
    ('ReLU', nn.ReLU(inplace=True)),
    ('split', torch.split(torch.nn.Dropout(p=0.25))),
    ('split_and_cat_dim_1', torch.split(torch.nn.Dropout2d(p=0.5)), torch.cat(torch.split(torch.nn.Dropout2d(p=0.5)), dim=1)),
    ('ReLU_1', nn.ReLU(inplace=True)),
    ('split2', torch.split(torch.nn.Dropout3d(p=0.75))),
    ('split_and_cat_dim_1_1', torch.split(torch.nn.Dropout2d(p=0.5)), torch.cat(torch.split(torch.nn.Dropout2d(p=0.5)), dim=1)),
    ('ReLU_2', nn.ReLU(inplace=True)),
]))
import torch.onnx
import itertools
# Input to the model
example_input = torch.randn(1, 3, 64, 128, 3).to('cpu')
# ONNX model
torch.onnx.export(model, example_input, ".", do_constant_folding=False, input_names=['input'], output_names=['output'])