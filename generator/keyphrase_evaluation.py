# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import datasets
import string
import numpy as np


_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""

BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class KeyphraseMetric(datasets.Metric):

    def _info(self):
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value('string')),
                'references': datasets.Sequence(datasets.Value('string')),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        return

    def _normalize_keyphrase(self, kp):

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_punc(lower(kp)))

    def _compute(self, predictions, references):
        """Returns the scores"""

        macro_metrics = {'precision': [], 'recall': [], 'f1': [], 'num_pred': [], 'num_gold': []}
        # macro_metrics = {'precision': [], 'recall': [], 'f1': []}

        # for context, targets, preds in zip(context_lines, target_lines, preds_lines):
        for targets, preds in zip(references, predictions):
            targets = [self._normalize_keyphrase(tmp_key).strip() for tmp_key in targets if len(self._normalize_keyphrase(tmp_key).strip()) != 0]
            preds = [self._normalize_keyphrase(tmp_key).strip() for tmp_key in preds if len(self._normalize_keyphrase(tmp_key).strip()) != 0]

            total_tgt_set = set(targets)
            total_preds = set(preds)
            if len(total_tgt_set) == 0: continue

            # get the total_correctly_matched indicators
            total_correctly_matched = len(total_preds & total_tgt_set)

            # macro metric calculating
            precision = total_correctly_matched / len(total_preds) if len(total_preds) else 0.0
            recall = total_correctly_matched / len(total_tgt_set)
            f1 = 2 * precision * recall / (precision + recall) if total_correctly_matched > 0 else 0.0
            macro_metrics['precision'].append(precision * 100.0)
            macro_metrics['recall'].append(recall * 100.0)
            macro_metrics['f1'].append(f1 * 100.0)
            macro_metrics['num_pred'].append(len(total_preds))
            macro_metrics['num_gold'].append(len(total_tgt_set))

        return {
            "precision": np.mean(macro_metrics["precision"]),
            "recall": np.mean(macro_metrics["recall"]),
            "f1": np.mean(macro_metrics["f1"]),
            "num_pred": np.mean(macro_metrics["num_pred"]),
            "num_gold": np.mean(macro_metrics["num_gold"]),
            "raw": macro_metrics,
        }

