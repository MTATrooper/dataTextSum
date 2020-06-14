from utils import _get_word_ngrams
import re
import json
from rouge import Rouge
import os
import collections
import pickle as pkl


Json_test_path = 'vietnews/test'
Json_train_path = 'vietnews/train'
Json_val_path = 'vietnews/val'
ref_path = 'vietnews/refs'

finished_files_dir = 'vietnews'
def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    # def _rouge_clean(s):
    #     return re.sub(r'[.\'\",]', '', s)
    abst = abstract_sent_list

    max_score = 0.0
    rouge = Rouge()
    selected = []
    for _ in range(summary_size):
        cur_id = -1
        cur_max_score = max_score
        for i in range(len(doc_sent_list)):
            if i in selected: continue
            c = selected + [i]
            line = ''
            for idx in c:
                line += doc_sent_list[idx] + ' '
            scores = rouge.get_scores(line, abst)
            if scores[0]['rouge-1']['r'] > cur_max_score:
                cur_max_score = scores[0]['rouge-1']['r']
                cur_id = i
        if cur_id == -1: break
        selected.append(cur_id)
        max_score = cur_max_score
    return selected

def create_json(src_path, des_path, makevocab = False, create_ref = False):
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    if makevocab:
        vocab_counter = collections.Counter()
    if create_ref:
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
    for filepath in os.listdir(src_path):
        try:
            with open(os.path.join(src_path, filepath), encoding='utf-8') as f:
                lines = f.readlines()
            data = {}
            data['title'] = ""
            data['abstract'] = ""
            data['article'] = []
            data['image'] = []
            tmp = 'title'
            space_count = 0
            for line in lines:
                if line == '\n': 
                    space_count += 1
                    continue
                if space_count == 0: tmp = 'title'
                elif space_count == 1: tmp = 'abstract'
                elif space_count == 2: tmp = 'article'
                else: tmp = 'image'
                if tmp == 'article' or tmp == 'image':
                    data[tmp].append(line.lower().strip())
                else: 
                    data[tmp] = line.lower().strip()
            doc_sent_list = data['article']
            abstract_sent_list = data['abstract']
            data['label'] = greedy_selection(doc_sent_list, abstract_sent_list, 3)
            outfile = filepath.split('/')[-1].split('.')[0]
            json.dump(data, open(os.path.join(des_path, '{}.json'.format(outfile)), 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
            if makevocab:
                art_tokens = ' '.join(doc_sent_list).split()
                abs_tokens = abstract_sent_list.split()
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t != ""] # remove empty
                vocab_counter.update(tokens)
            if create_ref:
                with open(os.path.join(ref_path, '{}.ref'.format(outfile)), 'w') as ref:
                    ref.write(data['abstract'])
        except:
            print('ERROR in file {}'.format(filepath))

    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
                  'wb') as vocab_file:
            pkl.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")

if __name__ == "__main__":
    print('Writing test file...')
    create_json('data/test_tokenized/', Json_test_path, create_ref = True)
    print('Writing val file...')
    create_json('data/val_tokenized/', Json_val_path)
    print('Writing train file...')
    create_json('data/train_tokenized/', Json_train_path, makevocab = True)
    