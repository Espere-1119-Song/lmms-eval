import os
import re
import unicodedata

import editdistance as ed  # TODO: new package
import zhconv  # TODO: new package

from lmms_eval.tasks.librispeech.cn_tn import TextNorm
from lmms_eval.tasks.librispeech.whisper_normalizer.basic import BasicTextNormalizer
from lmms_eval.tasks.librispeech.whisper_normalizer.english import EnglishTextNormalizer

# ImportError: To support decoding audio files, please install 'librosa' and 'soundfile'.
english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="",
)
basic_normalizer = BasicTextNormalizer()

dir_name = os.path.dirname(os.path.abspath(__file__))


def openasr_doc_to_audio(doc):
    return [doc["audio"]]


def openasr_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}Please recognize the speech and only output the recognized content:{post_prompt}"


def openasr_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""

    gt = doc["text"]

    data_dict = {"gt": gt, "pred": pred}

    return {"wer": data_dict}


PUNCS = "!,.?;:"


def remove_sp(text, language):
    gt = re.sub(r"<\|.*?\|>", " ", text)
    gt = re.sub(rf"\s+", r" ", gt)  # Replace consecutive spaces in the text with a single space.
    gt = re.sub(f" ?([{PUNCS}])", r"\1", gt)
    gt = gt.lstrip(" ")
    if language == "zh":
        gt = re.sub(rf"\s+", r"", gt)
    return gt


class EvaluationTokenizer(object):
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers
    in sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are
    applied after sacreBLEU tokenization.

    Args:
        tokenizer_type (str): the type of sacreBLEU tokenizer to apply.
        lowercase (bool): lowercase the text.
        punctuation_removal (bool): remove punctuation (based on unicode
        category) from text.
        character_tokenization (bool): tokenize the text to characters.
    """

    SPACE = chr(32)
    SPACE_ESCAPE = chr(9601)
    # ALL_TOKENIZER_TYPES = ChoiceEnum(["none", "13a", "intl", "zh", "ja-mecab"])

    def __init__(
        self,
        tokenizer_type: str = "13a",
        lowercase: bool = False,
        punctuation_removal: bool = False,
        character_tokenization: bool = False,
    ):
        # from sacrebleu.tokenizers import TOKENIZERS
        # from sacrebleu.tokenizers import tokenizer_none
        from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
        from sacrebleu.tokenizers.tokenizer_char import TokenizerChar
        from sacrebleu.tokenizers.tokenizer_intl import TokenizerV14International
        from sacrebleu.tokenizers.tokenizer_ja_mecab import TokenizerJaMecab
        from sacrebleu.tokenizers.tokenizer_none import NoneTokenizer
        from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

        TOKENIZERS = {
            "none": NoneTokenizer,
            "13a": Tokenizer13a,
            "intl": TokenizerV14International,
            "zh": TokenizerZh,
            "ja-mecab": TokenizerJaMecab,
            "char": TokenizerChar,
        }

        assert tokenizer_type in TOKENIZERS, f"{tokenizer_type}, {TOKENIZERS}"
        self.lowercase = lowercase
        self.punctuation_removal = punctuation_removal
        self.character_tokenization = character_tokenization
        self.tokenizer = TOKENIZERS[tokenizer_type]
        # self.tokenizer = tokenizer_none

    @classmethod
    def remove_punctuation(cls, sent: str):
        """Remove punctuation based on Unicode category."""
        return cls.SPACE.join(t for t in sent.split(cls.SPACE) if not all(unicodedata.category(c)[0] == "P" for c in t))

    def tokenize(self, sent: str):
        tokenized = self.tokenizer()(sent)

        if self.punctuation_removal:
            tokenized = self.remove_punctuation(tokenized)

        if self.character_tokenization:
            tokenized = self.SPACE.join(list(tokenized.replace(self.SPACE, self.SPACE_ESCAPE)))

        if self.lowercase:
            tokenized = tokenized.lower()

        return tokenized


def compute_wer(refs, hyps, language):
    distance = 0
    ref_length = 0
    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )
    for i in range(len(refs)):
        ref = refs[i]
        pred = hyps[i]
        if language in ["yue"]:
            ref = zhconv.convert(ref, "zh-cn")
            pred = zhconv.convert(pred, "zh-cn")
        if language in ["en"]:
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        if language in ["zh"]:
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)
        else:
            ref = basic_normalizer(ref)
            pred = basic_normalizer(pred)
        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()
        if language in ["zh", "yue"]:
            ref_items = [x for x in "".join(ref_items)]
            pred_items = [x for x in "".join(pred_items)]
        if i == 0:
            print(f"ref: {ref}")
            print(f"pred: {pred}")
            print(f"ref_items:\n{ref_items}\n{len(ref_items)}\n{ref_items[0]}")
            print(f"pred_items:\n{pred_items}\n{len(ref_items)}\n{ref_items[0]}")
        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    return distance / ref_length


def openasr_wer(results, args):
    # lan = args["language"]
    refs, hyps = [], []
    # results_list = results_dict[source]
    for result in results:
        lan = "en"
        gt = result["gt"]
        response = result["pred"]
        gt = remove_sp(gt, lan)
        response = remove_sp(response, lan)
        refs.append(gt)
        hyps.append(response)
    wer = compute_wer(refs, hyps, lan)
    # print(f"source: {source}  cnt: {len(refs)} wer: {wer:.4f}")
    return wer * 100

    # for gt, response, source, audio_path in zip(merged_gts, merged_responses, merged_sources, merged_audio_paths):
    #     results.append({
    #         'gt': gt,
    #         'response': response,
    #         'source': source,
    #         'audio_path': audio_path,
    #     })
    # time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    # results_file = f'{args.dataset}_{time_prefix}.json'
    # json.dump(results, open(results_file, 'w'))
    results_dict = {}
    for item in results:
        source = item["source"]
        results_dict.setdefault(source, []).append(item)
    lan = ds_collections[args.dataset]["language"]
    for source in results_dict:
        refs, hyps = [], []
        results_list = results_dict[source]
        for result in results_list:
            gt = result["gt"]
            response = result["response"]
            gt = remove_sp(gt, lan)
            response = remove_sp(response, lan)
            refs.append(gt)
            hyps.append(response)
        wer = compute_wer(refs, hyps, lan)
        print(f"source: {source}  cnt: {len(refs)} wer: {wer:.4f}")

    pass
