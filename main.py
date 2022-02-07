from data_processor import DataProcessor

if __name__ == "__main__":
    data_processor = DataProcessor()
    data_processor.load_batch_of_hands()

    dp = data_processor
    dp.batch_show()