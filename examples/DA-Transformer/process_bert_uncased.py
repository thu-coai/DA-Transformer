from transformers import BertTokenizer

def bert_uncased_tokenize(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in fin:
        word_pieces = tok.tokenize(line.strip())
        new_line = " ".join(word_pieces)
        fout.write('{}\n'.format(new_line))

bert_uncased_tokenize('train.src', 'tokenized_train.src')
bert_uncased_tokenize('train.tgt', 'tokenized_train.tgt')
bert_uncased_tokenize('dev.src', 'tokenized_dev.src')
bert_uncased_tokenize('dev.tgt', 'tokenized_dev.tgt')
bert_uncased_tokenize('test.src', 'tokenized_test.src')
bert_uncased_tokenize('test.tgt', 'tokenized_test.tgt')
