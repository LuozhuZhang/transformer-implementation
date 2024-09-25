from datasets import load_dataset

dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

for i in range(20):
  print(train_data[i]['text'])