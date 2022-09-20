# create pseudo positive labels for the training of DPR retriever
import re
import json
import os
from datasets import load_dataset, load_metric, concatenate_datasets
from collections import defaultdict
import csv
import random
from argparse import ArgumentParser
random.seed(9527)

parser = ArgumentParser()
parser.add_argument('--percent', default=2, type=int, help='# percent of alignment data in training set')
parser.add_argument('--version', default=1, type=int, help='t-iteration of iterative self training')
parser.add_argument('--retriever_ckpt', default=14, type=int, help='retriever checkpoints')
args = parser.parse_args()

data_path = '/home/ec2-user/quic-efs/user/yifangao/AmazonMultilingualKeyphraseDataset/'
en_data = {}
with open(os.path.join(data_path, f"en.json")) as f:
    for line in (f):
        ldata = json.loads(line)
        en_data[ldata['asin']] = ldata
print(f'Load {len(en_data)} us data')
en_asins = [k for k in en_data]

curr_percent, curr_version, curr_xdpr_ckpt = args.percent, args.version, args.retriever_ckpt

gold_nonalign_train_kps_path = f'/home/ec2-user/quic-efs/user/yifangao/AmazonMultilingualKeyphraseDataset/percent-{curr_percent}/mix.train.nonaligned.json'

pred_nonalign_train_kps_path = [
    f"/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_{curr_percent}/generator_v0/baseline_aligned/ddp_8_gpu_e10_lr1e-4/generations.mix.train.nonaligned.json",
    f"/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_{curr_percent}/generator_v{curr_version}/rakg_top5_aligned/ddp_8_gpu_e10_lr1e-4/generations.mix.train.nonaligned.json",
             ]
retrieved_kp_path = [
    f"/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_{curr_percent}/retriever_v{curr_version}_clean/dpr_biencoder.{curr_xdpr_ckpt}.us_0.retrieved_{curr_percent}_mix_train_dev_test_full",
]
print(pred_nonalign_train_kps_path[0])
print(pred_nonalign_train_kps_path[1])
print(retrieved_kp_path[0])

file_suffix = f"v{curr_version}_v0"

with open(os.path.join(gold_nonalign_train_kps_path)) as f:
    gold_nonalign_train_kps = [json.loads(line) for line in f]
with open(os.path.join(pred_nonalign_train_kps_path[-2])) as f:
    pred_nonalign_train_kps_old = json.load(f)
with open(os.path.join(pred_nonalign_train_kps_path[-1])) as f:
    pred_nonalign_train_kps_new = json.load(f)

# check gold and pred are ordered
for i in range(len(gold_nonalign_train_kps)):
    assert gold_nonalign_train_kps[i]['asin'] == pred_nonalign_train_kps_old['asin'][i] == pred_nonalign_train_kps_new['asin'][i]

kp_labels = [e['keywords'].split(';') for e in gold_nonalign_train_kps]
metric = load_metric("/home/ec2-user/quic-efs/user/yifangao/rakgP/keyphrase_evaluation.py")

result_old = metric.compute(predictions=pred_nonalign_train_kps_old['predictions'], references=kp_labels)
result_new = metric.compute(predictions=pred_nonalign_train_kps_new['predictions'], references=kp_labels)

# load retrieved kps
with open(retrieved_kp_path[-1]) as f:
    retrieved_kp = json.load(f)

topk = 1
threshold = 5
saving_path = pred_nonalign_train_kps_path[-1].replace("generations.mix.train.nonaligned.json", f"pseudo_label_topk_{topk}_threshold_{threshold}_{file_suffix}.json")
assert saving_path != pred_nonalign_train_kps_path[-1]

pseudo_positive_asin_ids = []
pseudo_positive_asin_ids_selected, pseudo_positive_asin_ids_nonselected = [], []
for asin_old, asin_new, f1_old, f1_new in zip(pred_nonalign_train_kps_old['asin'], pred_nonalign_train_kps_new['asin'], result_old['raw']['f1'], result_new['raw']['f1']):
    assert asin_old == asin_new
    asin_en = asin_old.split('_')[-1]
    positive_en_asin_predict = retrieved_kp[asin_old][0][:topk]
    if f1_new > f1_old + threshold:
        pseudo_positive_asin_ids.append(asin_old)
        pseudo_positive_asin_ids_selected.append(asin_en in positive_en_asin_predict)
    else:
        pseudo_positive_asin_ids_nonselected.append(asin_en in positive_en_asin_predict)

print(f'topk={topk}, threshold={threshold}, percent={curr_percent}, version={curr_version}, ckpt={curr_xdpr_ckpt}, '
      f'Creating {len(pseudo_positive_asin_ids_selected)}/{len(pseudo_positive_asin_ids_selected) + len(pseudo_positive_asin_ids_nonselected)} pseduo labels')

pseudo_training_data = []
asin2gold = {ex['asin']: ex for ex in gold_nonalign_train_kps}
for asin_id in pseudo_positive_asin_ids:
    positive_en_asin = retrieved_kp[asin_id][0][:topk]
    ex = asin2gold[asin_id]
    negative_asin = list(set(random.sample(en_asins, 100+len(positive_en_asin))) - set(positive_en_asin))

    new_ex = {
        'asin': ex['asin'],
        'question': ex['content'],
        'answer': [en_data[id]['content'] for id in positive_en_asin],
        'positive_ctxs': [{"text": en_data[id]['content']} for id in positive_en_asin],
        'negative_ctxs': [{"text": en_data[id]['content']} for id in negative_asin[:50]],
        'hard_negative_ctxs': [{"text": en_data[id]['content']} for id in negative_asin[50:100]]
    }
    pseudo_training_data.append(new_ex)

with open(saving_path, 'w') as f:
    json.dump(pseudo_training_data, f)
print(f'Saving to {saving_path}')








