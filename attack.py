# BERT sentence classification model implement from transformers package
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# You should prepare your parameter under the path
tokenizer = BertTokenizer.from_pretrained('/content/drive/My Drive/Research/bert_hotflip/parameter/sst')
model = BertForSequenceClassification.from_pretrained('/content/drive/My Drive/Research/bert_hotflip/parameter/sst')

# Global variable to record the grad
embedding_grad = None
now_length = None

# hook the grad during backwards
def hook_fn_backward(module, grad_input, grad_output):
  global embedding_grad
  embedding_grad = grad_output[0].squeeze()

# hook the gradient
modules = model.named_children()
for name, module in modules:
    if(name == 'bert'):
      submodules = module.named_children()
      for subname, submodule in submodules:
        if(subname == 'embeddings'):
          subsubmodules = submodule.named_children()
          for subsubname, subsubmodule in subsubmodules:
            if(subsubname == 'word_embeddings'):
              subsubmodule.register_backward_hook(hook_fn_backward)

# prepare your data, you can change the code according to the format of your data
data_path = '/content/drive/My Drive/Research/bert_hotflip/data/attack_sst.txt'
data = []
with open(data_path, 'r') as f:
  for line in f:
    example = line.strip().split(' ')
    data.append((' '.join(example[1:]), int(example[0])))
    # example = line.strip().split("\"")
    # data.append((' '.join(example[3:]), int(example[1])-1))

#original embedding
ori_emb = model.bert.embeddings.word_embeddings.weight

''' 
One step attack

n = len(data)
ok = 0
ok_attack = 0

for sentence, label in data:
  tokens = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)
  now_length = tokens.shape[1]
  truth = torch.tensor([label]).unsqueeze(0)
  loss = model(tokens, labels=truth)
  loss[0].backward()

  #english word 1996 to 29611 and 1037 to 1062, total 27642
  tokens = tokens.squeeze()
  delta_loss = []
  delta_tok = []
  for i in range(now_length):
    # print('original word ', tokenizer.decode([tokens[i]]))

    tok_emb = model.bert.embeddings.word_embeddings.weight[int(tokens[i])].repeat(30522, 1)
    delta = ori_emb - tok_emb
    tok_grad = embedding_grad[i].squeeze().unsqueeze(1)

    result = torch.mm(delta, tok_grad)
    delta_loss.append(torch.max(result))
    delta_tok.append(torch.argmax(result))
    # print('change to', tokenizer.decode([torch.argmax(result)]))
    # print('word token', torch.max(result))
  
  delta_loss = torch.tensor(delta_loss)
  delta_tok = torch.tensor(delta_tok)
  change_place = torch.argmax(delta_loss)
  # print('change place: ', change_place)
  # print('delta loss: ', torch.max(delta_loss))
  # print('orginal word: ', tokenizer.decode([tokens[change_place]]))
  # print('new word: ', tokenizer.decode([delta_tok[change_place]]))
  new_tokens = tokens.clone()
  new_tokens[change_place] = delta_tok[change_place]
  new_loss = model(new_tokens.unsqueeze(0), labels=truth)
  # print('original loss: ', loss[0])
  # print('new loss: ', new_loss[0])
  print('original prob: ', loss[1])
  print('new prob: ', new_loss[1])
  print('')

  if truth == torch.argmax(loss[1]):
    ok += 1
    if(truth != torch.argmax(new_loss[1])):
      ok_attack += 1

  # break

print('original accuracy: ', float(ok)/n)
print('attack accuracy: ', float(ok_attack)/n)

'''


# mulitstep attack
n = len(data)
ok = 0
ok_attack = 0
len_sum = 0
replace_sum = 0
f = open("result_sst.txt", "w")

for sentence, label in data:
  replace_times = 0
  tokens = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)
  now_length = tokens.shape[1]
  truth = torch.tensor([label]).unsqueeze(0)

  if truth != torch.argmax(model(tokens, labels=truth)[1]):
    continue
  
  ok += 1
  while(True):
    loss = model(tokens, labels=truth)
    loss[0].backward()
    tokens = tokens.squeeze()

    if(replace_times == now_length):
      print('unsuccessful')
      break

    if truth != torch.argmax(loss[1]):
      ok_attack += 1
      print(sentence)
      print(tokenizer.decode(tokens.squeeze(), skip_special_tokens = True))
      print('successful attack ', ok_attack)
      f.write(sentence+'\n')
      f.write(tokenizer.decode(tokens.squeeze(), skip_special_tokens = True)+'\n')
      f.write('\n')
      f.flush()
      len_sum += now_length
      replace_sum += replace_times
      break

    delta_loss = []
    delta_tok = []
    for i in range(now_length):
      tok_emb = model.bert.embeddings.word_embeddings.weight[int(tokens[i])].repeat(30522, 1)
      delta = ori_emb - tok_emb
      tok_grad = embedding_grad[i].squeeze().unsqueeze(1)
      result = torch.mm(delta, tok_grad)
      delta_loss.append(torch.max(result))
      delta_tok.append(torch.argmax(result))
    
    delta_loss = torch.tensor(delta_loss)
    delta_tok = torch.tensor(delta_tok)
    change_place = torch.argmax(delta_loss)
    tokens[change_place] = delta_tok[change_place]
    tokens = tokens.unsqueeze(0)
    replace_times += 1

f.close()
print('original accuracy: ', float(ok)/n)
print('attack accuracy: ', float(ok_attack)/n)
print('replace rate: ', float(replace_sum/len_sum))
