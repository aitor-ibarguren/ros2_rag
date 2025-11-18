import os
import sys
import unittest

if __name__ == '__main__' or __package__ is None:
    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )


from faiss_wrapper.faiss_wrapper import FAISSWrapper


class TestFAISSWrapper(unittest.TestCase):
    def test_init_new_index(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Init index again
        self.assertFalse(faiss_wrapper.init_new_index())

    def test_add_to_index(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Create text list
        text_list: list[str] = []
        text_list.append('Test message')
        text_list.append('Another test message')
        # Add to uninitialized index
        self.assertFalse(faiss_wrapper.add_to_index(text_list))
        # Check size of uninitialized index
        res, size = faiss_wrapper.get_index_size()
        self.assertTrue(not res and size == -1)
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add to index again
        self.assertTrue(faiss_wrapper.add_to_index(text_list))
        # Add to index with chunking
        text_list.clear()
        long_string = (
            "This is a long string that is split across multiple lines"
            " to verify the chunking capabilities of the FAISSWrapper class."
            "The chunker should divide the string into different chunks."
        )
        text_list.append(long_string)
        self.assertTrue(faiss_wrapper.add_to_index(text_list, True, 25, 5))
        # Check size
        res, size = faiss_wrapper.get_index_size()
        self.assertTrue(res and size == 11)

    def test_add_from_csv(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Add CSV to non-initialized index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertFalse(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'fake.csv')
        self.assertFalse(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Add CSV to index again
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertFalse(faiss_wrapper.add_from_csv(
            file_path, 'no-field'))
        # Add CSV to index one last time
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Check size
        res, size = faiss_wrapper.get_index_size()
        self.assertTrue(res and size == 15)

    def test_add_from_csv_2(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data', chunking=True, chunk_size=50,
            chunk_overlap=5))
        # Check size
        res, size = faiss_wrapper.get_index_size()
        self.assertTrue(res and size == 28)

    def test_add_pdfs_from_folder(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Add PDFs to non-initialized index
        current_dir = os.path.dirname(__file__)
        folder_path = os.path.join(current_dir, 'docs')
        self.assertFalse(faiss_wrapper.add_pdfs_from_folder(
            folder_path))
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add PDFs from non-existent folder
        current_dir = os.path.dirname(__file__)
        folder_path = os.path.join(current_dir, 'fake')
        self.assertFalse(faiss_wrapper.add_pdfs_from_folder(
            folder_path))
        # Add PDFs to index
        current_dir = os.path.dirname(__file__)
        folder_path = os.path.join(current_dir, 'docs')
        self.assertTrue(faiss_wrapper.add_pdfs_from_folder(
            folder_path))

    def test_add_web_search_DDG(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Add web search to non-initialized index
        self.assertFalse(faiss_wrapper.add_web_search_DDG(
            'History of Large Language Models (LLM)', 10))
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add web search to index
        self.assertTrue(faiss_wrapper.add_web_search_DDG(
            'History of Large Language Models (LLM)', 10))

    def test_search(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Search in non-initialized index
        res, output_texts, distances = faiss_wrapper.search([
            'Where is located the shop?',
            'What kind of musical instruments can I find in the shop?'
        ], 5)
        self.assertFalse(res)
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Search and check size
        res, output_texts, distances = faiss_wrapper.search([
            'Where is located the shop?',
            'What kind of musical instruments can I find in the shop?'
        ], 5)
        self.assertTrue(res and len(output_texts) == 2 and len(distances) == 2)
        self.assertTrue(all(len(sublist) == 5 for sublist in output_texts))
        self.assertTrue(all(len(sublist) == 5 for sublist in distances))

    def test_save_load_index(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Save non-initialized index
        current_dir = os.path.dirname(__file__)
        self.assertFalse(faiss_wrapper.save_index(current_dir, 'test_index'))
        # Create text list
        text_list: list[str] = []
        text_list.append('Test message')
        text_list.append('Another test message')
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add to index
        self.assertTrue(faiss_wrapper.add_to_index(text_list))
        # Save index
        current_dir = os.path.dirname(__file__)
        self.assertTrue(faiss_wrapper.save_index(current_dir, 'test_index'))
        # Save index in protected path
        self.assertFalse(faiss_wrapper.save_index('/root', 'test_index'))
        # Create new index
        new_faiss_wrapper = FAISSWrapper()
        # Try loading non-existent index
        self.assertFalse(
            new_faiss_wrapper.load_stored_index(current_dir, 'fake_index')
        )
        # Try loading stored index
        self.assertTrue(
            new_faiss_wrapper.load_stored_index(current_dir, 'test_index')
        )
        # Try loading stored index again
        self.assertFalse(
            new_faiss_wrapper.load_stored_index(current_dir, 'test_index')
        )

    def test_save_load_index_2(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Save index
        self.assertTrue(faiss_wrapper.save_index(current_dir, 'test_index_2'))
        # Create new index
        new_faiss_wrapper = FAISSWrapper()
        # Try loading stored index
        self.assertTrue(
            new_faiss_wrapper.load_stored_index(current_dir, 'test_index_2')
        )

    def test_keyword_search(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Try loading stored index
        self.assertTrue(
            faiss_wrapper.search_by_keyword(
                ["Can I buy guitar strings and picks?",
                 "Have you got an online webstore?"]
            )
        )

    def test_hybrid_search(self):
        # Create wrapper
        faiss_wrapper = FAISSWrapper()
        # Init new index
        self.assertTrue(faiss_wrapper.init_new_index())
        # Add CSV to index
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'data', 'shop_data.csv')
        self.assertTrue(faiss_wrapper.add_from_csv(
            file_path, 'shop data'))
        # Try loading stored index
        self.assertTrue(
            faiss_wrapper.hybrid_search(
                ["Can I buy guitar strings and picks?",
                 "Have you got an online webstore?"]
            )
        )


if __name__ == '__main__':
    unittest.main()
