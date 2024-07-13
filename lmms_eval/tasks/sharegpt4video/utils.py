import os
import json
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from pathlib import Path

import yaml
import sys

from loguru import logger as eval_logger

dir_name = os.path.dirname(os.path.abspath(__file__))

SHAREGPT4VIDEO_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
hf_home = os.path.expanduser(hf_home)
base_cache_dir = os.path.expanduser(hf_home)


def sharegpt4video_test_doc_to_visual(doc):
    with open(Path(__file__).parent / "sharegpt4video_test.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["video_id"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    elif os.path.exists(video_path.replace("mp4", "mov")):
            video_path = video_path.replace("mp4", "mov")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def sharegpt4video_test_doc_to_text(doc, model_specific_prompt_kwargs=None):
    return model_specific_prompt_kwargs["prompt"]


def sharegpt4video_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""

    data_dict = {"answer": doc["caption"], "pred": pred, "video_id": doc["video_id"], "video_source": doc["video_source"]}

    return {f"sharegpt4video_{metric}": data_dict for metric in SHAREGPT4VIDEO_METRICS}



def sharegpt4video_aggregation_result(results, metric, args=None):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]  # , (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        stored_results.append({"image_id": result["video_id"], "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append({"image_id": result["video_id"], "caption": a, "video_id": idx})
            idx += 1
        dataset["images"].append({"video_id": result["video_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    sharegpt4video_result = coco.loadRes(stored_results)
    sharegpt4video_eval = COCOEvalCap(coco, sharegpt4video_result)

    imgIds = sharegpt4video_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = sharegpt4video_eval.coco.imgToAnns[imgId]
        res[imgId] = sharegpt4video_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    path = generate_submission_file("sharegpt4video_captions_val_results.json", args)

    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Results saved to {path}")

    return score


def sharegpt4video_bleu4(results, args=None):
    return sharegpt4video_aggregation_result(results, "Bleu_4", args)


def sharegpt4video_bleu3(results, args=None):
    return sharegpt4video_aggregation_result(results, "Bleu_3", args)


def sharegpt4video_bleu2(results, args=None):
    return sharegpt4video_aggregation_result(results, "Bleu_2", args)


def sharegpt4video_bleu1(results, args=None):
    return sharegpt4video_aggregation_result(results, "Bleu_1", args)


def sharegpt4video_meteor(results, args=None):
    return sharegpt4video_aggregation_result(results, "METEOR", args)


def sharegpt4video_rougel(results, args=None):
    return sharegpt4video_aggregation_result(results, "ROUGE_L", args)


def sharegpt4video_cider(results, args=None):
    return sharegpt4video_aggregation_result(results, "CIDEr", args)


def sharegpt4video_spice(results, args=None):
    return sharegpt4video_aggregation_result(results, "SPICE", args)


def sharegpt4video_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case sharegpt4video_passthrough), value: metric value
    """
    return {"sharegpt4video_passthrough": {"pred": result, "image_id": doc["image_id"]}}


def sharegpt4video_test_aggregation_result(results, args):
    stored_results = []
    for result in results:
        stored_results.append({"image_id": result["image_id"], "caption": result["pred"]})

    path = generate_submission_file("sharegpt4video_captions_test2014_alg_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored into {path}. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
