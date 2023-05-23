import os

def find_image_file(filename, search_path='.'):
    for root, dirnames, filenames in os.walk(search_path):
        if filename in filenames:
            yield root

# Example usage:
search_dir = '/path/to/search/directory'
filename_to_find = 'example.jpg'

for folder in find_image_file(filename_to_find, search_dir):
    print(folder)
