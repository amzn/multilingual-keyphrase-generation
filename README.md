# Multilingual Keyphrase Generation

Main Repo: https://github.com/Yifan-Gao/multilingual_keyphrase_generation

If you have any question, you can contact yifangao@amazon.com

Create your own environment:
```shell
# create conda env for retriever
pip install src/retriever/requirements.txt
python -m spacy download en_core_web_sm

# create conda env for generator
pip install src/generator/requirements.txt
python -m spacy download en_core_web_sm
```

## 1. Cross-lingual Dense Passage Retriever

code path
```shell
src/retriever/
```

#### 1.1 Training

```shell
cd src/retriever/
./cmd/xdpr_train.sh <percent-of-alignment-data> <iteration>
```

By default we set the `<percent-of-alignment-data>` as 2%. It is also possible to change it to larger values such as 3,5,10.

`<iteration>` denotes the i-th loop in our iterative self-training algorithm. Here it should be set as `1` since it is the first loop of retriever training.

After training, we take the best checkpoint `<xdpr-best-ckpt-clean>` for generate dense embeddings for all English ASIN passages.

#### 1.2 Generate Dense Embeddings for All English ASIN Passages

```shell
cd src/retriever/
./cmd/gen_emb_kp_clean.sh <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

It takes 3 hours to generate dense Embeddings for 3M English passages on a single machine with 8 x 40GB GPUs.

#### 1.3 Inference

First, we evaluate our trained model on the test set:

```shell
cd src/retriever/
./cmd/xdpr_eval_mix_test_clean.sh <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

Then, we use the same model to retrieve English passages for instances in train and dev set. The retrieved English passages will be used for retrieval-augmented keyphrase generation.

```shell
cd src/retriever/
./cmd/xdpr_inference_for_rakg.sh <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

#### 1.4 An Example to Train and Evaluate Retriever

For example, we train the retriever with 2% parallel data:
```shell
cd src/retriever/
./cmd/xdpr_train.sh 2 1
# assume the best checkpoint is received at epoch 14
./cmd/gen_emb_kp_clean.sh 2 1 14
./cmd/xdpr_eval_mix_test_clean.sh 2 1 14
./cmd/xdpr_inference_for_rakg.sh 2 1 14
```

## 2. mBART-based Keyphrase Generator

code path
```shell
src/generator/
```

#### 2.1 Train on Full Data

First, we need to extract associated keyphrases from retrieved English passages:

```shell
cd src/generator/
python ./cmd/create_rkp.py <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

`<iteration>` denotes the i-th loop in our iterative self-training algorithm. Here it should be set as `1` since it is the first loop of retriever training.

`<xdpr-best-ckpt-clean>` is the retriever best checkpoint in the above training. 

Then, we train the keyphrase generation model on the full dataset:

```shell
cd src/generator/
./cmd/generator_vt/rakg_train_fulldata.sh <percent-of-alignment-data> <iteration> 
```

#### 2.2 An Example to Train and Evaluate Keyphrase Generator

For example, we train the retriever with 2% parallel data:
```shell
cd src/generator/
# assume the best checkpoint is received at epoch 14
python ./cmd/create_rkp.py 2 1 14
./cmd/generator_vt/rakg_train_fulldata.sh 2 1
```

## 3. Retriever Iterative Self-Training

First, we need to train a keyphrase generation baseline without using any retrieved keyphrase knowledge.

```shell
cd src/generator/
./cmd/generator_v0/baseline_train_aligned.sh <percent-of-alignment-data> <iteration>
```

Here `<iteration>` = 0 because it is our generator baseline model (generator_v0).

Taking the best checkpoint of the trained generator model on the alignment data, we do inference on the non-aligned training set:

```shell
cd src/generator/
./cmd/generator_v0/inference_nonaligned_training_set.sh <percent-of-alignment-data> <iteration> <best-generator-ckpt>
```

BTW, if we train it on the full dataset, it will become our mBART baseline:

```shell
cd src/generator/
./cmd/generator_v0/baseline_train_fulldata.sh <percent-of-alignment-data> <iteration>
```

#### 3.1 Train Retriever on Psuedo Data

If it is the first iteration (t=1), this step should be skipped since there is no pseudo labelled data generated.

```shell
cd src/retriever/
./cmd/xdpr_train_noisy.sh <percent-of-alignment-data> <previous-iteration> <iteration>
```

#### 3.2 Train Retriever on Clean Data

We finetune on the clean (alignment) data based on the best checkpoint `<xdpr-best-ckpt-noisy>` received from the pseudo labelled training:

```shell
cd src/retriever/
./cmd/xdpr_train_clean.sh <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-noisy>
```

#### 3.3 Generate Dense Embeddings for All English ASIN Passages

```shell
cd src/retriever/
./cmd/gen_emb_kp_clean.sh <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

#### 3.4 Retriever Inference

First, we evaluate our trained model on the test set:

```shell
cd src/retriever/
./cmd/xdpr_eval_mix_test_clean.sh <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

Then, we use the same model to retrieve English passages for instances in train and dev set. The retrieved English passages will be used for retrieval-augmented keyphrase generation.

```shell
cd src/retriever/
./cmd/xdpr_inference_for_rakg.sh <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

#### 3.5 Gernerator Training

First, we need to extract associated keyphrases from retrieved English passages:

```shell
cd src/generator/
python ./cmd/create_rkp.py <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

Then, we train the keyphrase generation model on the aligned dataset:

```shell
cd src/generator/
./cmd/generator_vt/rakg_train_aligned.sh <percent-of-alignment-data> <iteration> 
```

Taking the best checkpoint of the trained generator model on the alignment data, we do inference on the non-aligned training set:

```shell
cd src/generator/
./cmd/generator_vt/inference_nonaligned_training_set.sh <percent-of-alignment-data> <iteration> <best-generator-ckpt>
```

#### 3.6 Creating Pseudo Parallel Passage Pairs

```shell
cd src/generator/
python ./cmd/create_xdpr_pseudo_labels_by_generator_score_hard_threshold.py <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
```

Then we need to add json path of the generated pseudo labells into `src/retriever/conf/datasets/kp_xdpr.yaml` by creating the following entry:
```yaml
mkp_<percent-of-alignment-data>_mix_train_nonaligned_pseudo_label_v<iteration>:
  _target_: dpr.data.biencoder_data.JsonQADataset
  file: "path/to/pseudo_label.json"
```

`<percent-of-alignment-data>` and `<iteration>` should be replaced by real values.

#### 3.7 An Example For Iterative Self-Training
Assume we do iterative self-training with 2% alignment data:

```shell
# train keyphrase generation baseline (G_0)
cd src/generator/
./cmd/generator_v0/baseline_train_aligned.sh 2 0
# assump the best checkpoint is received at step 260
./cmd/generator_v0/inference_nonaligned_training_set.sh 2 0 260

# Iterative Self-Training. At the iteration t (t = 1, 2, 3 ...):
cd src/retriever/
# If it is the first iteration (t=1), this step should be skipped since there is no pseudo labelled data generated.
./cmd/xdpr_train_noisy.sh 2 t-1 t
# assume the best checkpoint in the noisy data training is received at epoch 12
./cmd/xdpr_train_clean.sh 2 t 12
# assume the best checkpoint in the clean data training is received at epoch 14
./cmd/gen_emb_kp_clean.sh 2 t 14
./cmd/xdpr_eval_mix_test_clean.sh 2 t 14
./cmd/xdpr_inference_for_rakg.sh 2 t 14
cd src/generator/
python ./cmd/create_rkp.py 2 t 14
./cmd/generator_vt/rakg_train_aligned.sh 2 t
# assume the best checkpoint in the generation training is received at step 260
./cmd/generator_vt/inference_nonaligned_training_set.sh 2 t 260
python ./cmd/create_xdpr_pseudo_labels_by_generator_score_hard_threshold.py 2 t 14
# Then we need to add json path of the generated pseudo labells into `src/retriever/conf/datasets/kp_xdpr.yaml`
# and we can continue to the next iteration (t+1) until there is no improvement for the retrieval recall
```


#### 3.8 Train Full Keyphrase Generation Models

After N iterations of retriever self-training, we may find more iterations may not bring improvements on the retrieval recall.

At this point, we train the full retrieval-augmented keyphrase generation model taking the lastest retriever:

```shell
cd src/generator/
python ./cmd/create_rkp.py <percent-of-alignment-data> <iteration> <xdpr-best-ckpt-clean>
./cmd/generator_vt/rakg_train_aligned.sh <percent-of-alignment-data> <iteration> 
```

