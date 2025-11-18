import ast
import os
import random
import re
import unicodedata
from urllib.parse import urlparse

import faiss
import fitz  # For PyMuPDF
import numpy as np
import pandas
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scipy.sparse import csr_matrix, load_npz, save_npz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class FAISSWrapper:

    def __init__(self):
        self._model_name = 'all-MiniLM-L6-v2'
        self._index_init = False
        self._splitter = None

        # Output vars
        self._RED = '\033[91m'
        self._YELLOW = '\033[33m'
        self._RST = '\033[0m'

        self._user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (X11; Linux x86_64)",
        ]

        self._headers = {
            "User-Agent": random.choice(self._user_agents),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.google.com",
            "DNT": "1",
            "Connection": "keep-alive"
        }

        self._junk_patterns = [
            "cookie settings", "accept cookies", "sign up", "login", "terms of service",
            "privacy policy", "navigation", "menu", "footer", "all rights reserved", "©",
        ]

    def init_new_index(self, keyword_number: int = 3) -> bool:
        # Check if already initialized
        if self._index_init:
            print(self._RED + 'Index already initialized/loaded!' + self._RST)
            return False

        # Store keyword number value
        self._keyword_number = keyword_number
        # Init model
        self._model = SentenceTransformer(self._model_name)
        # Init index
        dimensions = self._model.get_sentence_embedding_dimension()
        print(f'Creating index with {dimensions} dimensions...')

        self._index = faiss.IndexHNSWFlat(dimensions, 16)
        self._texts = []
        self._tf_idf_vectorizer = None
        self._tfidf_matrix = None
        self._keywords = []

        self._index_init = True

        # Output
        print('Index created!')

        return True

    def _handle_read_index_error(self, error: Exception, path: str) -> bool:
        if isinstance(error, FileNotFoundError):
            msg = f"File not found at '{path}'"
        elif isinstance(error, PermissionError):
            msg = f"Permission denied for '{path}'"
        else:
            msg = f"Error loading FAISS index: {error}"

        print(self._RED + msg + self._RST)

        return False

    def load_stored_index(self, path: str, index_name: str,
                          keyword_number: int = 3) -> bool:
        # Check if already initialized
        if self._index_init:
            print(self._RED + 'Index already initialized/loaded!' + self._RST)
            return False

        # Store keyword number value
        self._keyword_number = keyword_number

        # Ensure the path ends with a '/'
        path = path if path.endswith('/') else path + "/"

        # Manage paths and file names
        complete_index_path = path + index_name + '.faiss'
        complete_raw_text_path = path + index_name + '.csv'
        complete_tf_idf_path = path + index_name + '.npz'

        # Load index
        print(f"Loading index '{index_name}' from '{path}'...")

        try:
            self._index = faiss.read_index(complete_index_path)
        except Exception as e:
            return self._handle_read_index_error(e, complete_index_path)

        index_dimension = self._index.d
        print(f'Index with {index_dimension} dimensions...')

        # Init embedding model
        self._model = SentenceTransformer(self._model_name)

        embedding_dimension = self._model.get_sentence_embedding_dimension()
        print(f'Embedding model with {embedding_dimension} dimensions...')

        # Check if index and model dimensions match
        if index_dimension != embedding_dimension:
            print(self._RED +
                  'Index and embedding model dimensions do not match' +
                  self._RST
                  )
            return False

        self._index_init = True

        # Manage CSV
        try:
            # Load raw text CSV
            csv_data = pandas.read_csv(complete_raw_text_path, sep=';')
            # Get text
            self._texts = csv_data['text'].tolist()
            # Get keywords
            csv_data['keywords'] = csv_data['keywords'].apply(ast.literal_eval)
            self._keywords = csv_data['keywords'].tolist()
        except Exception as e:
            print(self._RED + 'Error loading/parsing CSV file: ' + e + self._RST)
            return False

        # Check row numbers
        if self._index.ntotal != len(self._texts):
            print(self._RED +
                  "Index and raw data files' row number do not match" +
                  self._RST)
            return False

        # Load NPZ file
        try:
            self._tfidf_matrix = load_npz(complete_tf_idf_path)
        except Exception as e:
            print(self._RED + 'Error loading NPZ file: ' + e + self._RST)
            return False

        # Output
        print('Index and raw text files successfully loaded!')

        return True

    def _chunk_texts(self, texts: list[str], chunk_size: int,
                     chunk_overlap: int) -> list[str]:
        # Instantiate the splitter if necessary
        if (
            self._splitter is None or
            chunk_size != self._splitter.chunk_size or
            chunk_overlap != self._splitter.chunk_overlap
        ):
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        # Chunk texts
        all_chunks = []
        for text in texts:
            all_chunks.extend(self._splitter.split_text(text))

        return all_chunks

    def _extract_matrix_and_keywords(self, texts: list[str],
                                     top_n: int = 3
                                     ) -> tuple[csr_matrix, list[list[str]]]:
        # Init list
        keywords = []

        # Check if TF-IDF Vectorizer init
        if self._tf_idf_vectorizer is None:
            # Initialize TF-IDF Vectorizer
            self._tf_idf_vectorizer = TfidfVectorizer(stop_words='english')
        # Calculate matrix
        tfidf_matrix = self._tf_idf_vectorizer.fit_transform(texts)
        # Get words
        words = self._tf_idf_vectorizer.get_feature_names_out()
        # Loop in texts
        for doc_index, text in enumerate(texts):
            tf_idf_vector = tfidf_matrix[doc_index]
            sorted_idx = np.argsort(tf_idf_vector.data)[::-1][:top_n]
            keywords.append(
                [words[tf_idf_vector.indices[i]] for i in sorted_idx]
            )

        return tfidf_matrix, keywords

    def add_to_index(self, texts: list[str], chunking: bool = False,
                     chunk_size: int = 256, chunk_overlap: int = 25) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # If chunking, manage texts in list
        if chunking:
            # Chunk
            _texts = self._chunk_texts(texts, chunk_size, chunk_overlap)

            print(
                f"Original texts divided into {len(_texts)} chunks")
        else:
            _texts = texts

        # Texts into embedding
        embeddings = self._model.encode(_texts, convert_to_numpy=True)
        # Add to index
        self._index.add(embeddings)

        # Store original texts
        self._texts.extend(_texts)

        # Extract & store keywords
        tfidf_matrix, keywords = self._extract_matrix_and_keywords(self._texts)
        self._tfidf_matrix = tfidf_matrix
        self._keywords = keywords

        # Output
        print(
            f'Added {len(_texts)} embeddings - '
            f'A total of {self._index.ntotal} embeddings on index'
        )

        return True

    def _handle_read_csv_error(self, error: Exception, path: str) -> bool:
        if isinstance(error, FileNotFoundError):
            msg = f"File '{path}' not found"
        elif isinstance(error, pandas.errors.ParserError):
            msg = f"Failed to parse '{path}'"
        else:
            msg = f"Unexpected error: {error}"

        print(self._RED + msg + self._RST)

        return False

    def add_from_csv(self, csv_file_path: str, csv_field: str,
                     sep: str = ';', chunking: bool = False,
                     chunk_size: int = 256, chunk_overlap: int = 25) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # Load CSV
        print(f"Loading CSV file '{csv_file_path}'...")

        try:
            csv_data = pandas.read_csv(csv_file_path, sep=sep)
        except Exception as e:
            return self._handle_read_index_error(e, csv_file_path)

        # Get values of provided field
        try:
            texts = csv_data[csv_field].tolist()
        except KeyError:
            print(self._RED +
                  f"Column '{csv_field}' not found in CSV"+self._RST)
            return False

        # If chunking, manage texts in list
        if chunking:
            # Chunk
            texts = self._chunk_texts(texts, chunk_size, chunk_overlap)

            print(
                f"Original texts divided into {len(texts)} chunks")

        # Texts into embedding
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        # Add to index
        self._index.add(embeddings)

        # Store original texts
        self._texts.extend(texts)

        # Extract & store matrix and keywords
        tfidf_matrix, keywords = self._extract_matrix_and_keywords(self._texts)
        self._tfidf_matrix = tfidf_matrix
        self._keywords = keywords

        # Output
        print(
            f'Added {len(texts)} embeddings - '
            f'A total of {self._index.ntotal} embeddings on index'
        )

        return True

    def _get_files_by_extension(self, folder_path: str,
                                extension: str) -> bool:
        return [
            f for f in os.listdir(folder_path)
            if f.endswith(extension) and os.path.isfile(
                os.path.join(folder_path, f))
        ]

    def _load_pdf_as_string(self, file_path: str) -> str:
        complete_string = ""
        with fitz.open(file_path) as doc:  # automatically closes
            for page in doc:
                complete_string += page.get_text()

        return complete_string

    def add_pdfs_from_folder(self, folder_path: str, chunk_size: int = 512,
                             chunk_overlap: int = 50) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # Manage path
        if not folder_path.endswith(os.sep):
            folder_path = folder_path + os.sep

        # Get folder files with PDF extension
        try:
            file_list = self._get_files_by_extension(folder_path, 'pdf')
        except Exception as e:
            print(self._RED + f'An unexpected error occurred: {e}' + self._RST)
            return False

        # Get file content iterating on all PDFs
        file_content = []

        for file in tqdm(file_list, desc="Document loading"):
            # Output
            print(f"Loading data from {file}...")
            # Get file content as string
            file_content.append(self._load_pdf_as_string(folder_path + file))

        # Chunk file content
        chunks = self._chunk_texts(file_content, chunk_size,
                                   chunk_overlap)

        print(f"PDFs divided into {len(chunks)} chunks")

        # Texts into embedding
        print('Embedding chunks...')
        embeddings = self._model.encode(chunks, convert_to_numpy=True)
        # Add to index
        print('Adding chunks to index...')
        self._index.add(embeddings)

        # Store original texts
        self._texts.extend(chunks)

        # Extract & store matrix and keywords
        tfidf_matrix, keywords = self._extract_matrix_and_keywords(self._texts)
        self._tfidf_matrix = tfidf_matrix
        self._keywords = keywords

        # Output
        print(
            f'Added {len(chunks)} embeddings - '
            f'A total of {self._index.ntotal} embeddings on index'
        )

        return True

    def _check_wikipedia_page(self, url: str, bs: BeautifulSoup) -> bool:
        # Check URL domain
        parsed_url = urlparse(url)
        if "wikipedia.org" not in parsed_url.netloc:
            return False

        # Check Wikipedia markers
        title_tag = bs.find("title")
        if not title_tag or "Wikipedia" not in title_tag.text:
            return False

        return True

    def _check_copyright(self, url: str, bs: BeautifulSoup) -> bool:
        # Combine all footer text content
        footers = bs.find_all(['footer', 'div'], class_=re.compile("footer",
                                                                   re.I))
        footer_text = " ".join([f.get_text(separator=" ",
                                           strip=True) for f in footers])

        # Also scan the whole page text in case it's not in a footer
        page_text = bs.get_text(separator=" ", strip=True)

        # Define regex pattern for copyright
        pattern = re.compile(r"(©|\bCopyright\b).*?\d{4}", re.IGNORECASE)

        # Check footer first, then full page
        if pattern.search(footer_text):
            return True
        elif pattern.search(page_text):
            return True
        else:
            return False

    def _handle_ddgs_error(self, error: Exception) -> bool:
        if isinstance(error, requests.exceptions.RequestException):
            msg = f"Network error '{error}'"
        else:
            msg = f"Unexpected error: {error}"

        print(self._RED + msg + self._RST)

        return False

    # Junk removal helpers
    def _is_junk(self, chunk: str) -> bool:
        text_lower = chunk.lower()
        return any(pat in text_lower for pat in self._junk_patterns)

    def _is_low_info(self, chunk: str) -> bool:
        tokens = chunk.split()
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
        return unique_ratio < 0.3

    def _count_unicode_scripts(self, chunk: str) -> bool:
        scripts = set()
        for char in chunk:
            if char.isalpha():
                try:
                    script = unicodedata.name(char).split(' ')[0]
                    scripts.add(script)
                except ValueError:
                    continue
        return len(scripts)

    def _is_ui_junk(self, chunk: str) -> bool:
        tokens = chunk.split()
        short_word_ratio = sum(1 for t in tokens if len(t) <= 3) / len(tokens)
        capitalized_ratio = sum(1 for t in tokens if t.istitle()) / len(tokens)
        no_punct = chunk.count('.') + chunk.count(',') < 2
        return short_word_ratio > 0.3 and capitalized_ratio > 0.2 and no_punct

    def _clean_chunks(self, chunks: list[str]) -> list[str]:
        clean_chunks = []

        for chunk in tqdm(chunks, desc="Chunk cleaning"):
            if (
                not self._is_junk(chunk) and not self._is_low_info(chunk) and
                self._count_unicode_scripts(chunk) <= 2 and
                not self._is_ui_junk(chunk)
            ):
                clean_chunks.append(chunk)

        return clean_chunks

    def add_web_search_DDG(self, web_search_query: str, max_results: int = 5,
                           chunk_size: int = 512,
                           chunk_overlap: int = 50) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # Get web search results
        try:
            web_search_content = []
            with DDGS() as ddgs:
                web_search_results = ddgs.text(web_search_query,
                                               max_results=max_results)
                for result in tqdm(web_search_results,
                                   desc="Web page parsing"):
                    # Get URL
                    url = result['href']
                    # Get URL content
                    self._headers['User-Agent'] = random.choice(
                        self._user_agents)
                    url_content = requests.get(url, headers=self._headers)
                    # Get BeautifulSoup object
                    bs = BeautifulSoup(url_content.text, 'html.parser')
                    # Check if Wikipedia page or has copyright
                    if (
                        not self._check_wikipedia_page(url, bs) and
                        self._check_copyright(url, bs)
                    ):
                        print(
                            self._YELLOW +
                            'Skipping URL due to legal/ethical issues:' + url
                            + self._RST)
                        continue
                    # Find main/article & remove HTML marks
                    main_or_article = bs.find('main') or bs.find('article')
                    if main_or_article:
                        clean_url_content = main_or_article.get_text(
                            separator=" ", strip=True)
                    else:
                        print(
                            self._YELLOW +
                            "'main' or 'article' tags not found in "
                            + url + self._RST)
                        continue
                    # Additional text cleaning
                    # clean_url_content = re.sub(r'\n\s*\n+', '\n\n',
                    #                            clean_url_content)
                    # Store
                    web_search_content.append(clean_url_content)
                    # print(url + "\n" + clean_url_content)
        except Exception as e:
            self._handle_ddgs_error(e)

        # Check if any content retrieved
        if len(web_search_content) == 0:
            return False

        # Chunk file content
        chunks = self._chunk_texts(web_search_content, chunk_size,
                                   chunk_overlap)

        print(f'Web search information divided into {len(chunks)} chunks')

        # Clean chunks
        chunks = self._clean_chunks(chunks)

        print(f'Chunks after cleaning: {len(chunks)}')

        # Texts into embedding
        print('Embedding chunks...')
        embeddings = self._model.encode(chunks, convert_to_numpy=True)
        # Add to index
        print('Adding chunks to index...')
        self._index.add(embeddings)

        # Store original texts
        self._texts.extend(chunks)

        # Extract & store matrix and keywords
        tfidf_matrix, keywords = self._extract_matrix_and_keywords(self._texts)
        self._tfidf_matrix = tfidf_matrix
        self._keywords = keywords

        # Output
        print(
            f'Added {len(chunks)} embeddings - '
            f'A total of {self._index.ntotal} embeddings on index'
        )

        return True

    def get_index_size(self) -> tuple[bool, int]:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False, -1

        # Output
        return True, self._index.ntotal

    def search(
            self, texts: list[str],
            top_k: int = 5
    ) -> tuple[bool, list[list[str]], list[list[float]]]:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False, [], []

        # Texts into embedding
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        # Search
        distances, indices = self._index.search(embeddings, top_k)
        # Reconstruct
        output_texts = []
        for row in indices:
            row_texts = [self._texts[i] for i in row]
            output_texts.append(row_texts)

        return True, output_texts, distances

    def search_by_keyword(
            self, texts: list[str],
            top_k: int = 5
    ) -> tuple[bool, list[list[str]], list[list[float]]]:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False, [], []

        # Check if TF-IDF Vectorizer init
        if self._tf_idf_vectorizer is None:
            # Initialize TF-IDF Vectorizer
            self._tf_idf_vectorizer = TfidfVectorizer(stop_words='english')
        # Calculate matrix
        tfidf_matrix = self._tf_idf_vectorizer.transform(texts)

        # Cosine similarity
        similarity = cosine_similarity(tfidf_matrix, self._tfidf_matrix)

        # Loop in texts
        output_texts = []
        output_similarities = []
        for doc_index, text in enumerate(texts):
            similarity_vector = similarity[doc_index]
            sorted_idx = np.argsort(similarity_vector.data)[::-1][:top_k]
            # Iterate and remove zero similarity values
            row_texts = []
            row_similarities = []
            for i in sorted_idx:
                if similarity_vector[i] != 0:
                    row_texts.append(self._texts[i])
                    row_similarities.append(similarity_vector[i])
            # Insert in output list
            output_texts.append(row_texts)
            output_similarities.append(row_similarities)

        return True, output_texts, output_similarities

    def _normalize_vals(self, distances: list[float]):
        # Get min and max
        min_val, max_val = min(distances), max(distances)
        # Check if min and max are equal
        if max_val == min_val:
            return [0.0 for _ in distances]

        return [(val - min_val) / (max_val - min_val) for val in distances]

    def hybrid_search(
            self, texts: list[str],
            top_k: int = 5, alpha: int = 0.6
    ) -> tuple[bool, list[list[str]], list[list[float]]]:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False, [], []

        # Semantic search (top_k x 2 to ensure that a good combined search)
        res, semantic_texts, semantic_distances = self.search(texts, 2 * top_k)

        # Check result
        if not res:
            print(self._RED + 'Error in semantic search!' + self._RST)
            return False, [], []

        # Keyword search (top_k x 2 to ensure that a good combined search)
        res, keyword_texts, keyword_similarities = self.search_by_keyword(
            texts,
            2 * top_k
        )

        # Normalize distances
        normalized_semantic_distances = [self._normalize_vals(
            sublist) for sublist in semantic_distances]
        normalized_keyword_similarities = [self._normalize_vals(
            sublist) for sublist in keyword_similarities]

        # Invert similarity to compare with distances
        normalized_keyword_similarities_inv = [
            [1.0 - val for val in list]
            for list in normalized_keyword_similarities
        ]

        # Iterate over each search
        output_texts = []
        output_distances = []
        for st, kt, nsd, nksi in zip(semantic_texts, keyword_texts,
                                     normalized_semantic_distances,
                                     normalized_keyword_similarities_inv):
            # Create dictionaries
            semantic_dict = dict(zip(st, nsd))
            keyword_dict = dict(zip(kt, nksi))

            # Get all texts
            all_texts = set(semantic_dict) | set(keyword_dict)

            # Calculate combined distances
            combined_results = []
            for text in all_texts:
                # If text is not found, distance = 1.0
                semantic_dist = semantic_dict.get(text, 1.0)
                keyword_dist = keyword_dict.get(text, 1.0)
                combined_dist = alpha * semantic_dist + \
                    (1 - alpha) * keyword_dist
                combined_results.append({"text": text,
                                         "combined_distance": combined_dist})

            # Sort
            sorted_combined_distances = sorted(combined_results,
                                               key=lambda x:
                                               x["combined_distance"])[:top_k]

            # Extract & insert texts & distances
            output_texts.append([item["text"]
                                for item in sorted_combined_distances]
                                )
            output_distances.append(
                [item["combined_distance"]
                    for item in sorted_combined_distances]
            )

        return True, output_texts, output_distances

    def save_index(self, path: str, index_name: str) -> bool:
        # Check if already initialized
        if not self._index_init:
            print(self._RED + 'Index not initialized/loaded yet!' + self._RST)
            return False

        # Ensure the path ends with a '/'
        if not path.endswith('/'):
            path += '/'

        # Manage paths and file names
        complete_index_path = path + index_name + '.faiss'
        complete_raw_text_path = path + index_name + '.csv'
        complete_tf_idf_path = path + index_name + '.npz'

        # Save/write index
        try:
            faiss.write_index(self._index, complete_index_path)
        except PermissionError:
            print(self._RED + 'Permission denied: unable to write the index' +
                  self._RST)
            return False
        except IOError as e:
            print(self._RED + f'I/O error occurred while saving the index: {e}'
                  + self._RST)
            return False
        except Exception as e:
            print(self._RED + f'An unexpected error occurred: {e}' + self._RST)
            return False

        # Save raw text data
        df = pandas.DataFrame(self._texts, columns=['text'])
        df['keywords'] = self._keywords
        df.to_csv(complete_raw_text_path, sep=';',
                  index=True, index_label='ID')

        # Save TF-IDF matrix
        save_npz(complete_tf_idf_path, self._tfidf_matrix)

        # Output
        print('Index and raw text files successfully created!')

        return True
