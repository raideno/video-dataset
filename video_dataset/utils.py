def better_listdir(self, path):
    return list(filter(lambda file_name: file_name != ".DS_Store", os.listdir(path)))