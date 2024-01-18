class TextLoader:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load(self):
        loaded_data = []
        for file_path in self.file_paths:
            try:
                with open(file_path, 'r') as file:
                    text_data = file.read()
                    loaded_data.append(text_data)

            except Exception as e:
                print(f"Failed to load data from {file_path}. Error: {e}")

        return loaded_data

    @staticmethod
    def chunk_data(data, chunk_size):
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks
