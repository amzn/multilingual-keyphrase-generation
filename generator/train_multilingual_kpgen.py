#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np
import torch
from datasets import load_dataset, load_metric, concatenate_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import WEIGHTS_NAME
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from model import MultilingualKeyphraseGenerationModel
from trainer import MultilingualKeyphraseGenerationTrainer


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_path_for_inference: str = field(
        default=None,
        metadata={"help": "Path to trained model"}
    )
    model_path_for_pretreined_ckpt: str = field(
        default=None,
        metadata={"help": "Path to trained model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class KeyphraseArguments(DataTrainingArguments):
    """
    Arguments for multilingual keyphrase generation.
    """

    retrieval_augmented_generation: bool = field(
        default=False, metadata={"help": "use ground truth / retrieved english keyphrases for retrieval augmented generation"}
    )
    langs: Optional[str] = field(
        default="de-de_DE;es-es_XX;fr-fr_XX;it-it_IT",
        metadata={
            "help": "languages used for multilingual setting: <kp-dataset-lang-id>-<mBART-lang-id>;"
                    "mbart: ar_AR,cs_CZ,de_DE,en_XX,es_XX,"
                    "mbart: et_EE,fi_FI,fr_XX,gu_IN,hi_IN,"
                    "mbart: it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,"
                    "mbart: lv_LV,my_MM,ne_NP,nl_XX,ro_RO,"
                    "mbart: ru_RU,si_LK,tr_TR,vi_VN,zh_CN"
        },
    )
    retrieved_kp_file: Optional[str] = field(default=None, metadata={"help": "The retrieved english keyphrases {asin: kps}"})
    # max_kp_tokens: Optional[int] = field(default=512, metadata={"help": "max number of tokens for english keyphrase sequences"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, KeyphraseArguments, Seq2SeqTrainingArguments,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, = parser.parse_args_into_dataclasses()

    # parse input languages
    data_args.langs = [tuple(kv.split('-')) for kv in data_args.langs.split(";")]
    data_args.langs = {kv[0]: kv[1] for kv in data_args.langs}

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if data_args.train_file is not None:
        if "mix" not in data_args.train_file:
            raise NotImplementedError
            data_files["train"] = [data_args.train_file.format(lang) for lang in data_args.langs]
        else:
            data_files["train"] = [data_args.train_file]
    if data_args.validation_file is not None:
        # data_files["validation"] = [data_args.validation_file.format(lang) for lang in data_args.langs]
        if "mix" not in data_args.validation_file:
            raise NotImplementedError
            data_files["validation"] = [data_args.validation_file.format(lang) for lang in data_args.langs]
        else:
            data_files["validation"] = [data_args.validation_file]
    if data_args.test_file is not None:
        # data_files["test"] = [data_args.test_file.format(lang) for lang in data_args.langs]
        if "mix" not in data_args.test_file:
            raise NotImplementedError
            data_files["test"] = [data_args.test_file.format(lang) for lang in data_args.langs]
        else:
            data_files["test"] = [data_args.test_file]
    datasets = load_dataset('json', data_files=data_files)

    # load aligned / retrieved english keyphrases
    if data_args.retrieval_augmented_generation:
        assert data_args.retrieved_kp_file is not None, "retrieved_kp_file is needed for retrieval augmented generation!"
        with open(data_args.retrieved_kp_file) as f:
            retrieved_keyphrase_mapping = json.load(f)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # update max length and beam for config (because the current evaluate pipeline in trainer does not pass any args to it)
        max_length=data_args.max_target_length,
        num_beams=data_args.num_beams,
    )
    # add <kp-sep> as segmenter between keyphrases
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = MultilingualKeyphraseGenerationModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    num_added_toks = tokenizer.add_tokens(['<kp-sep>', '<context>', '<En-kps>', '<title>'])
    logger.info(f'We have added {num_added_toks} tokens: {num_added_toks}')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))


    # Yifan: here we do not define the `config.decoder_start_token_id` because we decode in a multilingual setting
    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            data_args.target_lang is not None and data_args.source_lang is not None
        ), "mBart requires --target_lang and --source_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if data_args.source_lang is not None:
            tokenizer.src_lang = data_args.source_lang
        if data_args.target_lang is not None:
            tokenizer.tgt_lang = data_args.target_lang

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        kp_sep_token = '<kp-sep>'
        en_kps_token = '<En-kps>'
        kp_sep_token_id = tokenizer.vocab[kp_sep_token]
        if data_args.retrieval_augmented_generation:
            targets = [kp_sep_token.join(ex.split(';')) for ex in examples["keywords"]]
            passage_inputs = examples["content"]
            passage_inputs_t = tokenizer(passage_inputs, max_length=data_args.max_source_length, padding=False, truncation=True)

            retrieved_keyphrases = [retrieved_keyphrase_mapping[ex_asin].replace(';', kp_sep_token) + kp_sep_token for ex_asin in examples["asin"]]
            retrieved_keyphrases_t = tokenizer(retrieved_keyphrases, max_length=data_args.max_source_length, padding=False, truncation=True, add_special_tokens=False)
            # crop according to max length of bart (1024)
            cropped_keyphrases = []
            for i in range(len(passage_inputs)):
                # 1 token for <En-kps>, so 1023 is the max length
                curr_max_len = max(1023 - len(passage_inputs_t.data['input_ids'][i]), 64)
                curr_retrieved_keyphrases_ids_cropped = retrieved_keyphrases_t.data['input_ids'][i][:curr_max_len]
                while curr_retrieved_keyphrases_ids_cropped[-1] != kp_sep_token_id and len(curr_retrieved_keyphrases_ids_cropped):
                    curr_retrieved_keyphrases_ids_cropped = curr_retrieved_keyphrases_ids_cropped[:-1]
                curr_retrieved_keyphrases_wds_cropped = tokenizer.decode(curr_retrieved_keyphrases_ids_cropped)
                cropped_keyphrases.append(curr_retrieved_keyphrases_wds_cropped)
            # <En-kps> en_kp1 <kp-sep> en_kp2 ... <kp-sep> en_kpN <kp-sep> <title> xxx <context> xxx <eos> <lang_id>
            inputs = [en_kps_token + r_k + inp for r_k, inp in zip(cropped_keyphrases, passage_inputs)]
            assert len(inputs) == len(targets)
        else:
            inputs = examples["content"]
            targets = [kp_sep_token.join(ex.split(';')) for ex in examples["keywords"]]

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        # reset current <lang_id> to the correct one, we allow examples with different lang ids
        for i in range(len(inputs)):
            cur_lang = data_args.langs[examples['lang'][i]]
            if isinstance(tokenizer, MBartTokenizer):
                cur_lang_id = tokenizer.lang_code_to_id[cur_lang]
            else:
                cur_lang_id = tokenizer.convert_tokens_to_ids(cur_lang)
            model_inputs['input_ids'][i][-1] = cur_lang_id
            labels['input_ids'][i][-1] = cur_lang_id

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        # add lang & asin
        model_inputs["lang"] = examples['lang']
        model_inputs["asin"] = examples['asin']
        return model_inputs


    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = train_dataset.shuffle(training_args.seed)
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric : replace it with customized metric
    metric = load_metric("./keyphrase_evaluation.py")

    def postprocess_text(preds, labels):
        preds = [[kp.strip() for kp in kps.split('<kp-sep>')] for kps in preds]
        labels = [[kp.strip() for kp in kps.split('<kp-sep>')] for kps in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)

        return result

    def compute_language_specific_metrics(raw_results, aligned_langs, langs):
        """
        Compute language specific results: Fr, De, Es, It ...
        @:param raw_results: raw results for individual samples
        @:param aligned_langs: associated lang for each individual sample
        @:param langs: available languages for compute
        """
        language_specific_result = {}
        for lang in langs:
            curr_p, curr_r, curr_f = [], [], []
            for raw_precision, raw_recall, raw_f1, aligned_lang in zip(
                    raw_results['precision'], raw_results['recall'], raw_results['f1'], aligned_langs):
                if aligned_lang == lang:
                    curr_p.append(raw_precision)
                    curr_r.append(raw_recall)
                    curr_f.append(raw_f1)
            if len(curr_p):
                language_specific_result[f"{lang}_precision"] = np.mean(curr_p)
                language_specific_result[f"{lang}_recall"] = np.mean(curr_r)
                language_specific_result[f"{lang}_f1"] = np.mean(curr_f)
        return language_specific_result

    # Initialize our Trainer
    trainer = MultilingualKeyphraseGenerationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        if model_args.model_path_for_pretreined_ckpt is not None:
            if os.path.isfile(os.path.join(model_args.model_path_for_pretreined_ckpt, WEIGHTS_NAME)):
                logger.info(f"Loading model from {model_args.model_path_for_pretreined_ckpt}).")
                if isinstance(trainer.model, PreTrainedModel):
                    trainer.model = trainer.model.from_pretrained(model_args.model_path_for_pretreined_ckpt)
                else:
                    state_dict = torch.load(os.path.join(model_args.model_path_for_pretreined_ckpt, WEIGHTS_NAME))
                    trainer.model.load_state_dict(state_dict)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        # compute language specific metrics
        language_specific_results = compute_language_specific_metrics(metrics['eval_raw'], eval_dataset['lang'], data_args.langs)
        metrics.update({"eval_"+k: v for k, v in language_specific_results.items()})
        del metrics['eval_raw']
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        # load trained ckpts (for inference)
        if training_args.do_train is False:
            resume_from_checkpoint = model_args.model_path_for_inference
            if resume_from_checkpoint is not None and os.path.isfile(
                    os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                logger.info(f"Loading model from {resume_from_checkpoint}).")
                if isinstance(trainer.model, PreTrainedModel):
                    trainer.model = trainer.model.from_pretrained(resume_from_checkpoint)
                else:
                    state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
                    trainer.model.load_state_dict(state_dict)

            # If model was re-initialized, put it on the right device and update trainer.model_wrapped
            if trainer.place_model_on_device:
                trainer.model = trainer.model.to(trainer.args.device)
            trainer.model_wrapped = trainer.model

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = test_results.metrics
        # compute language specific metrics
        language_specific_results = compute_language_specific_metrics(metrics['test_raw'], test_dataset['lang'], data_args.langs)
        metrics.update({"test_"+k: v for k, v in language_specific_results.items()})
        del metrics['test_raw']
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        test_log_file_name = data_args.test_file.split('/')[-1].replace('.json', '')
        trainer.log_metrics(test_log_file_name, metrics)
        trainer.save_metrics(test_log_file_name, metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                decoded_preds = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True)
                decoded_preds = [[kp.strip() for kp in kps.split('<kp-sep>')] for kps in decoded_preds]
                test_dataset_dict = test_dataset.to_dict()
                test_dataset_dict["predictions"] = decoded_preds
                test_file_name = data_args.test_file.split('/')[-1]
                output_test_preds_file = os.path.join(training_args.output_dir, "generations."+test_file_name)
                logger.info(f"saving predictions to {output_test_preds_file}")
                with open(output_test_preds_file, "w") as f:
                    json.dump(test_dataset_dict, f)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()