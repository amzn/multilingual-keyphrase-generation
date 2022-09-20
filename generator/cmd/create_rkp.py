import json
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--percent', default=2, type=int, help='# percent of alignment data in training set')
parser.add_argument('--version', default=1, type=int, help='t-iteration of iterative self training')
parser.add_argument('--retriever_ckpt', default=14, type=int, help='retriever checkpoints')
args = parser.parse_args()

# load english keyphrases
us_asin2kps = {}
with open(
        f'/home/ec2-user/quic-efs/user/yifangao/AmazonMultilingualKeyphraseDataset/en.json') as f:
    for line in (f):
        ex = json.loads(line)
        us_asin2kps[ex['asin']] = ex['keywords']
print(f'Load {len(us_asin2kps)} us raw data')

retrieved_kp_path = "/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_{}/retriever_v{}_clean/dpr_biencoder.{}.us_0.retrieved_{}_mix_train_dev_test_full"
output_kp_path = "/home/ec2-user/quic-efs/user/yifangao/SearchQURetrievalKeyphraseGeneration/ckpts/percent_{}/generator_v{}"

curr_percent, curr_version, curr_xdpr_ckpt = args.percent, args.version, args.retriever_ckpt

curr_retrieved_kp_path = retrieved_kp_path.format(curr_percent, curr_version, curr_xdpr_ckpt, curr_percent)
curr_output_kp_path = output_kp_path.format(curr_percent, curr_version)
print(curr_retrieved_kp_path)
print(curr_output_kp_path)

os.makedirs(curr_output_kp_path, exist_ok=True)

# load retrieved kps
with open(curr_retrieved_kp_path) as f:
    retrieved_kp = json.load(f)

# load multilingual keyphrases
langs = ['de', 'fr', 'es', 'it']
lang_ids = []
for lang in langs:
    for split in ['train', 'dev', 'test']:
        lang_data = []
        filename = ""
        train_dev_test_path = f"/percent-{curr_percent}" if split == 'train' else ""
        with open(
                f'/home/ec2-user/quic-efs/user/yifangao/AmazonMultilingualKeyphraseDataset{train_dev_test_path}/{lang}.{split}{filename}.json') as f:
            for line in (f):
                lang_data.append(json.loads(line))
        print(f'Load {len(lang_data)} {lang} {split} data')
        dpr_data = []
        for ex in lang_data:
            ex_asin = ex['asin']
            lang_ids.append((ex_asin, lang, split))

assert len(lang_ids) == len(retrieved_kp)
for top_k in [5,]:
    retrieved_mapping = {}
    for id in lang_ids:
        ex_asin, lang, split = id
        rkp = retrieved_kp[ex_asin]
        top_rkps = [us_asin2kps[kp] for kp in rkp[0][:top_k]]
        top_rkps = ";".join(top_rkps)
        if ex_asin in retrieved_mapping:
            print('debug')
        else:
            retrieved_mapping[ex_asin] = top_rkps
    assert len(retrieved_mapping) == len(retrieved_kp)
    saving_kp_path = os.path.join(curr_output_kp_path, f"asin2keyphrase.top{top_k}.json")
    print(f"saving to {saving_kp_path}")
    with open(saving_kp_path, 'w') as f:
        json.dump(retrieved_mapping, f)

