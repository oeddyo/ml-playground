import unittest
from translation_data import TranslationData


class TestTranslationData(unittest.TestCase):

    def setUp(self):
        self.translation_data = TranslationData(min_freq=1, max_sentence_len=200, src_lang="en", dest_lang="zh")

    def test_get_vocab(self):
        src_vocab, dest_vocab = self.translation_data.get_vocab()
        self.assertIsNotNone(src_vocab)
        self.assertIsNotNone(dest_vocab)
        self.assertTrue(len(src_vocab) > 0)
        self.assertTrue(len(dest_vocab) > 0)

    def test_get_datasets(self):
        training_set, validation_set = self.translation_data.get_datasets()
        self.assertIsNotNone(training_set)
        self.assertIsNotNone(validation_set)
        self.assertTrue(len(training_set) > 0)
        self.assertTrue(len(validation_set) > 0)

    def test_tokenization(self):
        example = {
            "translation": {
                "en": "This is a test sentence.",
                "zh": "这是一个测试句子。"
            }
        }
        tokenized_example = self.translation_data._tokenize(example)
        self.assertIn("src_tokens", tokenized_example)
        self.assertIn("dest_tokens", tokenized_example)
        self.assertTrue(len(tokenized_example["src_tokens"]) > 0)
        self.assertTrue(len(tokenized_example["dest_tokens"]) > 0)

    def test_vectorization(self):
        example = {
            "src_tokens": ["<SOS>", "This", "is", "a", "test", "sentence", ".", "<EOS>"],
            "dest_tokens": ["<SOS>", "这", "是", "一", "个", "测试", "句子", "。", "<EOS>"]
        }
        vectorized_example = self.translation_data._vectorize(example)
        self.assertIn("src_ids", vectorized_example)
        self.assertIn("dest_ids", vectorized_example)
        self.assertTrue(len(vectorized_example["src_ids"]) > 0)
        self.assertTrue(len(vectorized_example["dest_ids"]) > 0)


if __name__ == '__main__':
    unittest.main()
