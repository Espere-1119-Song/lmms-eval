import os
import zipfile

def get_files_in_directory(directory, extension=".mp4"):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def get_file_size(file):
    return os.path.getsize(file)

def create_zip_chunk(zip_name, files):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))

def chunk_files(files, chunk_size):
    chunks = []
    current_chunk = []
    current_size = 0

    for file in files:
        file_size = get_file_size(file)
        if current_size + file_size > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0
        current_chunk.append(file)
        current_size += file_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def main(source_directory, target_directory, chunk_size_gb=30):
    chunk_size = chunk_size_gb * (1024 ** 3)  # Convert GB to bytes
    files = get_files_in_directory(source_directory)
    chunks = chunk_files(files, chunk_size)

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for i, chunk in enumerate(chunks, 1):
        zip_name = os.path.join(target_directory, f"videos_chunked_{i:02d}.zip")
        create_zip_chunk(zip_name, chunk)
        print(f"Created {zip_name} with {len(chunk)} files.")

if __name__ == "__main__":
    source_directory = os.path.expanduser('~/.cache/huggingface/sharegpt4video')
    target_directory = os.path.expanduser('~/data/ShareGPT4Video')
    main(source_directory, target_directory)
