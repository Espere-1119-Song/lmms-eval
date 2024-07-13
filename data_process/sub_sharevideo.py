# import json
# from collections import Counter

# data_path = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video-raw/sharegpt4video_40k.jsonl'

# video_sources = []

# with open(data_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         data = json.loads(line.strip())
#         video_source = data['video_path'].split('/')[0]
#         video_sources.append(video_source)

# # 使用Counter统计每个video_source的数量
# source_counts = Counter(video_sources)

# # 打印结果
# for source, count in source_counts.items():
#     print(f'{source}: {count}')

import json
import random
from collections import Counter, defaultdict

data_path = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video-raw/sharegpt4video_40k.jsonl'
output_path = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video-raw/sharegpt4video_100_sampled.jsonl'

video_sources = defaultdict(list)

# 读取数据并按video_source分类
with open(data_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        video_source = data['video_path'].split('/')[0]
        video_sources[video_source].append(data)

# 计算每个子集需要采样的数据量
total_samples = 100
source_counts = Counter({source: len(items) for source, items in video_sources.items()})
total_items = sum(source_counts.values())
sample_counts = {source: int(total_samples * count / total_items) for source, count in source_counts.items()}

# 从每个子集中采样
sampled_data = []
for source, count in sample_counts.items():
    sampled_data.extend(random.sample(video_sources[source], count))

# 将采样的数据保存为jsonl文件
with open(output_path, 'w', encoding='utf-8') as out_file:
    for data in sampled_data:
        json.dump(data, out_file)
        out_file.write('\n')

print(f'Sampled {len(sampled_data)} items and saved to {output_path}')
