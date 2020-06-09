#hotflip to ESIM
import re
import torch
import pickle
from esim.model import ESIM

#prepare your parameter
checkpoint = torch.load('/content/drive/My Drive/Research/bert_hotflip/parameter/qqp/best.pth.tar', map_location='cuda:0')
# Retrieving model parameters from checkpoint.
# vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
# embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
vocab_size = checkpoint['model']['word_embedding.weight'].size(0)
embedding_dim = checkpoint['model']['word_embedding.weight'].size(1)
hidden_size = checkpoint['model']['projection.0.weight'].size(0)
num_classes = checkpoint['model']['classification.6.weight'].size(0)

# print(vocab_size)
# print(embedding_dim)
# print(hidden_size)
# print(num_classes)

print("\t* Building model...")
model = ESIM(vocab_size,
          embedding_dim,
          hidden_size,
          num_classes=num_classes,
          device='cuda').to('cuda')

model.load_state_dict(checkpoint['model'])

worddict = []
worddict_path = '/content/drive/My Drive/Research/bert_hotflip/parameter/ESIM/worddict.pkl'
with open(worddict_path, 'rb') as pkl:
  worddict = pickle.load(pkl)

indexdict = {value:key for key, value in worddict.items()}

embedding_grad = None
now_length = None

def hook_fn_backward(module, grad_input, grad_output):
  global embedding_grad
  global now_length
  # print(module)
  # print(len(grad_input))
  # print(len(grad_output))
  p = grad_output[0].squeeze()
  q = p.shape[0]
  if q == now_length:
    embedding_grad = p

# hook the gradient
modules = model.named_children()
for name, module in modules:
    if(name == 'word_embedding'):
      module.register_backward_hook(hook_fn_backward)

# data_path = '/content/drive/My Drive/Research/bert_hotflip/data/attack_snli.txt'
data_path = '/content/drive/My Drive/Research/bert_hotflip/data/attack_qqp.txt'
data = []
# labelMap = {"entailment":0, "neutral":1, "contradiction":2}
with open(data_path, 'r') as f:
  for line in f:
    example = line.strip().split('\t')
    # print(example)
    # if example[0] in labelMap:
      # data.append((labelMap[example[0]], example[1], example[2]))
    data.append((int(example[0]), example[1], example[2]))

biaodian = [',','.','?','!',':']

def words_to_indices(sentence):
    indices = []
    indices.append(worddict["_BOS_"])
    for word in sentence.strip().split(' '):
        if word[-1] in biaodian:
          word1 = "".join(word[:-1])
          word2 = word[-1]

          if word1 in worddict:
            index = worddict[word1]
          else:
            index = worddict['_OOV_']
          indices.append(index)

          if word2 in worddict:
            index = worddict[word2]
          else:
            index = worddict['_OOV_']
          indices.append(index)

        else:
          if word in worddict:
            index = worddict[word]
          else:
            index = worddict['_OOV_']
          indices.append(index)
        
    indices.append(worddict["_EOS_"])
    return indices

def indices_to_words(indice, hypothesis):
    p = hypothesis.strip().split(' ')
    q = []
    for word in p:
      if word[-1] in biaodian:
        q.append(''.join(word[:-1]))
        q.append(word[-1])
      else:
        q.append(word)

    sen = []
    for i in range(len(indice)):
      if indice[i] <=3:
        sen.append(q[i])
      else:
        sen.append(indexdict[indice[i]])
    
    return(" ".join(sen))

n = len(data)
ok = 0
ok_attack = 0
len_sum = 0
replace_sum = 0
ori_emb = model.word_embedding.weight
f = open("result_snli.txt", "w")

for label, premise, hypothesis in data:
  replace_times = 0
  # print(premise)
  # print(hypothesis)

  p_tokens = torch.tensor([words_to_indices(premise)]).to('cuda')
  p_length = torch.tensor([p_tokens.shape[1]]).to('cuda')
  h_tokens = torch.tensor([words_to_indices(hypothesis)]).to('cuda')
  h_length = torch.tensor([h_tokens.shape[1]]).to('cuda')
  
  truth = torch.tensor(label).to('cuda')
  _, prob = model(p_tokens, p_length, h_tokens, h_length)
  result = torch.argmax(prob)

  if truth != result:
    continue
  
  ok += 1
  now_length = h_length
  while(True):
    _, prob = model(p_tokens, p_length, h_tokens, h_length)
    print(prob)
    prob.backward(torch.ones(1,2).to('cuda'))
    h_tokens = h_tokens.squeeze()

    if(replace_times == now_length - 2):
      print('unsuccessful')
      break

    if truth != torch.argmax(prob):
      ok_attack += 1
      print(hypothesis)
      print(indices_to_words(h_tokens.tolist()[1:-1], hypothesis))
      print('successful attack ', ok_attack)
      f.write(premise+'\n')
      f.write(hypothesis+'\n')
      f.write(indices_to_words(h_tokens.tolist()[1:-1], hypothesis)+'\n')
      f.write('\n')
      f.flush()
      len_sum += now_length
      replace_sum += replace_times
      break

    delta_loss = []
    delta_tok = []
    print(h_tokens)
    for i in range(1, now_length-1):
      if indexdict[int(h_tokens[i])] in biaodian:
        continue
      # tok_emb = model._word_embedding.weight[int(h_tokens[i])].repeat(42394, 1)
      tok_emb = model.word_embedding.weight[int(h_tokens[i])].repeat(42394, 1)
      delta = ori_emb - tok_emb
      tok_grad = embedding_grad[i].squeeze().unsqueeze(1)
      result = torch.mm(delta, tok_grad)
      delta_loss.append(torch.max(result))
      delta_tok.append(torch.argmax(result))
    
    delta_loss = torch.tensor(delta_loss)
    delta_tok = torch.tensor(delta_tok)
    change_place = torch.argmax(delta_loss)
    h_tokens[change_place + 1] = delta_tok[change_place]
    h_tokens = h_tokens.unsqueeze(0)
    replace_times += 1

f.close()
print('original accuracy: ', float(ok)/n)
print('attack accuracy: ', float(ok_attack)/n)
print('replace rate: ', float(replace_sum/len_sum))
