import json
import os
import shutil

json_file_path = 'data/ShareGPT4Video/sharegpt4video_mix181k_vqa-153k_share-cap-28k.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

destination_folder = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video/instruction_videos/panda'
error_names = []
error_json_file = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video/error_files.json'

for meta_data in data:
    image_path = meta_data['image']
    try:
        # if 'bdd/' in image_path:
        #     video_folder = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video/zip_folder/bdd100k'
        #     video_name = image_path.split('/')[1].split('_')[0] + '.mov'
        #     video_path =  os.path.join(video_folder, video_name)
        #     destination_file = os.path.join(destination_folder, video_name)
        #     shutil.move(video_path, destination_file)
        #     print(f'File moved to {destination_file}')

        # if 'clevrer_qa/' in image_path:
        #     video_name = meta_data['id']
        #     video_id = int(video_name.split('_')[1])
        #     if video_id < 1000:
        #         video_folder = '/data/wenhao/data/video_00000-01000'
        #     elif video_id < 2000:
        #         video_folder = '/data/wenhao/data/video_01000-02000'
        #     elif video_id < 3000:
        #         video_folder = '/data/wenhao/data/video_02000-03000'
        #     elif video_id < 4000:
        #         video_folder = '/data/wenhao/data/video_03000-04000'
        #     elif video_id < 5000:
        #         video_folder = '/data/wenhao/data/video_04000-05000'
        #     elif video_id < 6000:
        #         video_folder = '/data/wenhao/data/video_05000-06000'
        #     elif video_id < 7000:
        #         video_folder = '/data/wenhao/data/video_06000-07000'
        #     elif video_id < 8000:
        #         video_folder = '/data/wenhao/data/video_07000-08000'
        #     elif video_id < 9000:
        #         video_folder = '/data/wenhao/data/video_08000-09000'
        #     else:
        #         video_folder = '/data/wenhao/data/video_09000-10000'
        #     video_name = 'video_'+str(video_id).zfill(5) + '.mp4'
        #     video_path =  os.path.join(video_folder, video_name)
        #     destination_file = os.path.join(destination_folder, video_name)
        #     if not os.path.exists(destination_file):
        #         shutil.move(video_path, destination_file)
        #         print(f'File moved to {destination_file}')

        # if 'ego4d/' in image_path:
        #     video_folder = '/data/wenhao/projects/lmms-eval/data/ShareGPT4Video/zip_folder/ego4d'
        #     video_name = image_path.split('/')[1].split('_')[0] + '.mp4'
        #     video_path =  os.path.join(video_folder, video_name)
        #     destination_file = os.path.join(destination_folder, video_name)
        #     if os.path.exists(video_path):
        #         shutil.move(video_path, destination_file)
        #         print(f'File moved to {destination_file}')


        # if 'videochatgpt/' in image_path:
        #     video_folder = '/data/wenhao/data/ActivityNetQA/all_test/'
        #     video_id = meta_data['id']
        #     video_name = video_id + '.mp4'
        #     video_path =  os.path.join(video_folder, video_name)
        #     if os.path.exists(video_path):
        #         destination_file = os.path.join(destination_folder, video_name)
        #         shutil.move(video_path, destination_file)
        #         print(f'File moved to {destination_file}')
        #     else:
        #         video_name = video_id + '.mkv'
        #         video_path =  os.path.join(video_folder, video_name)
        #         if os.path.exists(video_path):
        #             destination_file = os.path.join(destination_folder, video_name)
        #             shutil.move(video_path, destination_file)
        #             print(f'File moved to {destination_file}')

        # if 'next_qa/' in image_path:
        #     video_folder = '/data/wenhao/data/NExTVideo'
        #     video_id = meta_data['image']
        #     video_name = video_id.split('.')[0].split('next_qa/')[-1] + '.mp4'
        #     video_path =  os.path.join(video_folder, video_name)
        #     destination_file = os.path.join(destination_folder, video_name.split('/')[1])
        #     if os.path.exists(video_path):
        #         shutil.move(video_path, destination_file)
        #         print(f'File moved to {destination_file}')

        if 'panda/' in image_path:
            video_folder = '/data/wenhao/data/ShareGPT4Video/zip_folder/panda'
            video_id = meta_data['id']
            video_name = video_id + '.mp4'
            video_path =  os.path.join(video_folder, video_name)
            destination_file = os.path.join(destination_folder, video_name)
            if os.path.exists(video_path):
                shutil.move(video_path, destination_file)
                print(f'File moved to {destination_file}')


    except:
        import pdb;pdb.set_trace()
        error_names.append(image_path)


with open(error_json_file, 'a') as json_file:
    json.dump(error_names, json_file, indent=4)
