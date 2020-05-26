import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
model, vocab  = get_pytorch_kobert_model()
sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)
print("finish")
#pooled_output.shape
#torch.Size([2, 768])
#>>> vocab
#Vocab(size=8002, unk="[UNK]", reserved="['[MASK]', '[SEP]', '[CLS]']")
#>>> # Last Encoding Layer
#>>> sequence_output[0]
#tensor([[-0.2461,  0.2428,  0.2590,  ..., -0.4861, -0.0731,  0.0756],
#        [-0.2478,  0.2420,  0.2552,  ..., -0.4877, -0.0727,  0.0754],
#        [-0.2472,  0.2420,  0.2561,  ..., -0.4874, -0.0733,  0.0765]],
#       grad_fn=<SelectBackward>)
