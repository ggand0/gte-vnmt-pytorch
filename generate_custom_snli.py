import os, sys
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

# ref: https://github.com/pytorch/text/blob/master/torchtext/datasets/snli.py
batch_size = 128
inputs = data.Field(lower=False)
answers = data.Field(sequential=False)
train, dev, test = datasets.SNLI.splits(inputs, answers)
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=batch_size, device=0)

from collections import defaultdict
word_count = defaultdict(float)
for i in train:
    #print(i.premise)
    #print(i.hypothesis)
    for j in i.premise:
        word_count[j] += 1
    for j in i.hypothesis:
        word_count[j] += 1
print( 'word count: %d' % len(list(word_count.keys())) )


print('Extracting entailment pairs...')
# Create a simplified dataset containing the entailment label only
import os, json

root_path = '.data/snli/snli_1.0_entail'
train = 'snli_1.0_train.jsonl'
dev = 'snli_1.0_dev.jsonl'
test = 'snli_1.0_test.jsonl'
train_new = 'snli_1.0_train_entail.jsonl'
dev_new = 'snli_1.0_dev_entail.jsonl'
test_new = 'snli_1.0_test_entail.jsonl'

org_files = [train, dev, test]
new_files = [train_new, dev_new, test_new]

for index, org_file in enumerate(org_files):
    print('#'*50)
    with open(os.path.join(root_path, org_file)) as f:
        lines = f.readlines()
        print(lines[0])

        lines_new = []
        for i, line in enumerate(lines):
            if i % 100000 == 0:
                print(i)
            data = json.loads(line)
            if data['gold_label'] == 'entailment':
                lines_new.append(line)

        print(lines_new[100])
        print(len(lines_new))
        with open(os.path.join(root_path, new_files[index]), 'w') as f_new:
            #f_new.write("\n".join(lines_new))
            f_new.write("".join(lines_new))
print('DONE.')



import re, collections, operator, pickle, json

def tokenize(sent):
    '''
    data_reader.tokenize('a#b')
    ['a', '#', 'b']
    '''
    return [x.strip().lower() for x in re.split('(\W+)?', sent) if x.strip()]

root_path = '.data/snli/snli_1.0_entail'
train_new = 'snli_1.0_train_entail.jsonl'
dev_new = 'snli_1.0_dev_entail.jsonl'
test_new = 'snli_1.0_test_entail.jsonl'
new_files = [train_new, dev_new, test_new]

word_counts = collections.defaultdict(float)

for index, new_file in enumerate(new_files):
    print('#'*50)
    with open(os.path.join(root_path, new_file)) as f:
        lines = f.readlines()
        print(lines[1])

        lines_new = []
        for i, line in enumerate(lines):
            if i % 100000 == 0:
                print(i)
            data = json.loads(line)
            if data['gold_label'] == 'entailment':
                tokens1 = tokenize(data['sentence1'])
                tokens2 = tokenize(data['sentence2'])
                for token in tokens1:
                    word_counts[token] += 1
                for token in tokens2:
                    word_counts[token] += 1
                #lines_new.append(line)


print( len(list(word_counts.keys())) )
sorted_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_counts[:100])
with open(os.path.join(root_path, 'word_counts.dat'), 'wb') as f:
    pickle.dump(sorted_counts, f)
print('DONE.')



print('Removing duplicates...')
import fileinput
with open(os.path.join(root_path, 'word_counts.dat'), 'rb') as f:
    word_count = pickle.load(f)
print( len(list(word_counts.keys())) )

sorted_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=False)
print(sorted_counts[:50])


UNK_TOKEN = 'UNK'
# ref: https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file-using-python
for index, new_file in enumerate(new_files):
    print('#'*50)

    with open(os.path.join(root_path, new_file), 'r') as file :
        #filedata = file.read()
        lines = file.readlines()

    lines_reduced = []
    for i, line in enumerate(lines):
        if i % 100 == 0:
            print(i)
            sys.stdout.flush()

        data = json.loads(line)

        #print(type(json.dumps(data)))

        for j, token in enumerate(sorted_counts[:10000]):
            tokens1 = tokenize(data['sentence1'])
            tokens2 = tokenize(data['sentence2'])
            #if token[0] in tokens1 or token[0] in tokens2:
            #    data['sentence1'] = data['sentence1'].replace(token[0], UNK_TOKEN)
            #    data['sentence2'] = data['sentence2'].replace(token[0], UNK_TOKEN)


            replaced1 = [UNK_TOKEN if x!=UNK_TOKEN and token[0] == x else x for x in tokens1]
            replaced2 = [UNK_TOKEN if x!=UNK_TOKEN and token[0] == x else x for x in tokens2]
            data['sentence1'] = " ".join(replaced1)
            data['sentence2'] = " ".join(replaced2)
            #if token[0] in tokens1 or token[0] in tokens2:
            #    print(replaced1)
            #    print(replaced2)
            #    print(data['sentence1'])
            #    print(data['sentence2'])


        lines_reduced.append(json.dumps(data))

    with open(os.path.join(root_path, '%s_reduced'%new_file), 'w') as f:
        f.write("\n".join(lines_reduced))

        # Replace the target string
    #   filedata = filedata.replace(token[0], UNK_TOKEN)
    #   Write the file out again
    #with open(os.path.join(root_path, '%s_reduced'%new_file), 'w') as file:
    #    file.write(filedata)