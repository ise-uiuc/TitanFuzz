"""
prompts
"""
input_tensor = torch.randn(1, 1, 3, 3)
input_tensor = torch.randn(1, 1, 3, 3)
padding = 1
"none"
output_tensor = input_tensor
output_tensor = input_tensor
if True:
    output_tensor = torch.nn.ConstantPad2d(padding, 0)(input_tensor)
