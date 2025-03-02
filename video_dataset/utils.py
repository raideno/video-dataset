import os

def better_listdir(path):
    return list(filter(lambda file_name: file_name != ".DS_Store", os.listdir(path)))