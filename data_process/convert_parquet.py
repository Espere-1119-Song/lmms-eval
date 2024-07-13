# import pandas as pd
# import json
# import os
# import sys

# filtered_data = []
# cache_dir = '/data/wenhao/.cache/huggingface/sharegpt4video_test'
# jsonl_path = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video-raw/sharegpt4video_100_sampled.jsonl'

# with open(jsonl_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         record = json.loads(line)
#         video_id = record['video_id']
#         print(video_id)
#         video_name = record['video_path']
#         video_source = video_name.split('/')[0]
#         video_path = video_id + ".mp4"
#         video_path = os.path.join(cache_dir, video_path)
#         if os.path.exists(video_path):
#             video_path = video_path
#         elif os.path.exists(video_path.replace("mp4", "MP4")):
#             video_path = video_path.replace("mp4", "MP4")
#         elif os.path.exists(video_path.replace("mp4", "mkv")):
#             video_path = video_path.replace("mp4", "mkv")
#         elif os.path.exists(video_path.replace("mp4", "mov")):
#             video_path = video_path.replace("mp4", "mov")
#         else:
#             print(f"video path:{video_name} does not exist, please check")
#             continue
        
#         caption = record['captions'][-1]['content']
#         filter_record = {'video_id': video_id, 'caption': caption, 'video_source': video_source}
#         filtered_data.append(filter_record)
# print(len(filtered_data))
# df = pd.DataFrame(filtered_data)
# df.to_parquet('/data/wenhao/projects/lmms-eval/data/ShareGPT4Video/sharegpt4video_test/test-00000-of-00001.parquet')


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

input_file = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video/sharegpt4video_test/test-00000-of-00001.parquet'
output_file = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video/sharegpt4video_test/test-00000-of-00001.parquet'

table = pq.read_table(input_file)

df = table.to_pandas()

df_selected = df[['video_id', 'caption', 'video_source']]

df_selected['caption'] = df_selected['caption'].apply(lambda x: x if isinstance(x, list) else [x])

table_selected = pa.Table.from_pandas(df_selected)

pq.write_table(table_selected, output_file)

print("Finished processing and saving the parquet file.")
