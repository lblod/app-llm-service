from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import numpy as np
import networkx as nx
from library.BPMNGraph import BPMNGraph
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import GlorotUniform, Zeros
from spektral.layers import GCNConv, GlobalAvgPool, GlobalAttentionPool
import os
import logging
from library.chunking import TextChunker

# Define the custom objects
custom_objects = {
    'GCNConv': GCNConv,
    'GlobalAvgPool': GlobalAvgPool,
    'GlobalAttentionPool': GlobalAttentionPool,
    'GlorotUniform': GlorotUniform,
    'Zeros': Zeros
}

class GraphProcessor:
    """
    A base class for processing graphs.
    """

    def get_node_features(self, graph):
        """
        Extracts the features of the nodes in the given graph.

        Parameters:
        graph (networkx.Graph or BPMNGraph): The graph from which to extract node features.

        Returns:
        numpy.ndarray: A 2D array where each row represents the features of a node.
        """
        # Get node features
        nxgraph = graph if isinstance(graph, nx.Graph) else graph.get_nx_graph()
        x = [node.get("embedding", np.zeros(384)).reshape(-1) for _, node in nxgraph.nodes(data=True)]
        return np.stack(x)

    def get_adjacency_matrix(self, graph):
        """
        Computes the adjacency matrix of the given graph.

        Parameters:
        graph (networkx.Graph or BPMNGraph): The graph for which to compute the adjacency matrix.

        Returns:
        scipy.sparse.coo.coo_matrix: The adjacency matrix of the graph.
        """
        # Get adjacency matrix
        nxgraph = graph if isinstance(graph, nx.Graph) else graph.get_nx_graph()
        return nx.adjacency_matrix(nxgraph).tocoo()

class TextEmbedder:
    """
    A class for embedding text and keyword extraction.
    """

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        print(f"Loading sentence model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Loading keybert model: {model_name}")
        self.key_bert_model = KeyBERT(model_name)

    def extract_keywords(self, text, min_text_length=10, max_keyphrase_length=1, threshold=0.5):
        try:
            if not text or len(text) < min_text_length:
                return []       
            doc_embedding, word_embeddings = self.key_bert_model.extract_embeddings(text, keyphrase_ngram_range=(1, max_keyphrase_length))
            keywords = [keyword for keyword in self.key_bert_model.extract_keywords(text, doc_embeddings=doc_embedding, word_embeddings=word_embeddings) if keyword[1] > threshold]
            return keywords
        except Exception as e:
            logging.warning(f"Failed to extract keywords from text: '{text}'. Likely due to text length being too short.")
            return []

    def extract_document_embedding(self, text):
        cleaned_text = text.strip()
        return self.model.encode([cleaned_text])[0]

    def update_keyword_frequency(self, keywords, freq_dict):
        for keyword in keywords:
            freq_dict[keyword[0]] = freq_dict.get(keyword[0], 0) + 1
        return freq_dict

    def chunk_text(self, text, chunk_size=256):
        """
        Splits the text into chunks of a certain size.
    
        Parameters:
        text (str): The text to split.
        chunk_size (int): The size of the chunks.
    
        Returns:
        list of str: The text chunks.
        """
        return TextChunker.chunk_by_max_words(text, chunk_size)
    
    def process_text(self, text, min_text_length=1, max_keyphrase_length=1, threshold=0.5, chunk_size=512):
        """
        Processes a text string, extracting keywords and embeddings.
    
        Parameters:
        text (str): The text to process.
        min_text_length (int): The minimum length of text to consider when extracting keywords.
        max_keyphrase_length (int): The maximum length of keyphrases to consider when extracting keywords.
        threshold (float): The minimum score for a keyword to be considered.
        chunk_size (int): The size of the chunks to split the text into.
    
        Returns:
        dict: A dictionary mapping keywords to their frequencies in the text.
        list of dict: A list of dictionaries, each containing an index, a chunk of text, and its embedding.
        """
        chunks = self.chunk_text(text, chunk_size=chunk_size)
        freq_dict = {}
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            keywords  = self.extract_keywords(chunk, min_text_length=min_text_length, max_keyphrase_length=max_keyphrase_length, threshold=threshold)
            freq_dict = self.update_keyword_frequency(keywords, freq_dict)
            doc_embedding = self.extract_document_embedding(chunk)
            chunk_embeddings.append({"index": i, "chunked_text": chunk, "embedding": doc_embedding})
        return freq_dict, chunk_embeddings

class BPMNGraphEmbedder(TextEmbedder):
    """
    A base class for embedding BPMN graphs.
    """

    def get_embedding_model(self):
        return self.model

    def get_node_text(self, node, text_attributes):
        node_text = ''
        for field in text_attributes:
            node_text += node.get(field, '') + ' '
        return node_text.strip()

    def generate_bpmn_embedding(self, graph):
        embeddings = []
        for _, node in graph.nodes(data=True):
            if 'embedding' in node and node['embedding'] is not None:
                embeddings.append(node['embedding'])
        return np.mean(embeddings, axis=0).reshape(-1)

    def _extract_keywords_and_update_embeddings(self, nxgraph, text_attributes, min_text_length, max_keyphrase_length, threshold):
        """
        Helper method to extract keywords and update node embeddings for a NetworkX graph.

        Parameters:
        nxgraph (networkx.Graph): The graph to process.
        text_attributes (list of str): The attributes to use when extracting text from nodes.
        min_text_length (int): The minimum length of text to consider when extracting keywords.
        max_keyphrase_length (int): The maximum length of keyphrases to consider when extracting keywords.
        threshold (float): The minimum score for a keyword to be considered.

        Returns:
        dict: A dictionary mapping keywords to their frequencies in the graph.
        """
        graph_freq_keywords = {}
        for _, node in nxgraph.nodes(data=True):
            node_text = self.get_node_text(node, text_attributes=text_attributes)
            if not node_text or len(node_text) < min_text_length:
                continue
            keywords = self.extract_keywords(node_text, min_text_length=min_text_length, max_keyphrase_length=max_keyphrase_length, threshold=threshold)
            graph_freq_keywords = self.update_keyword_frequency(keywords, graph_freq_keywords)
            doc_embedding = self.extract_document_embedding(node_text)
            node['embedding'] = doc_embedding
        return graph_freq_keywords    

    def process_graph_nx(self, nxgraph, text_attributes=['name', 'documentation'], min_text_length=1, max_keyphrase_length=1, threshold=0.5, logging=False):
        """
        Processes a NetworkX graph, extracting keywords and embeddings for its nodes.

        Parameters:
        nxgraph (networkx.Graph): The graph to process.
        text_attributes (list of str): The attributes to use when extracting text from nodes.
        min_text_length (int): The minimum length of text to consider when extracting keywords.
        max_keyphrase_length (int): The maximum length of keyphrases to consider when extracting keywords.
        threshold (float): The minimum score for a keyword to be considered.
        logging (bool): Whether to log progress.
        """
        graph_freq_keywords = self._extract_keywords_and_update_embeddings(nxgraph, text_attributes, min_text_length, max_keyphrase_length, threshold)
        nxgraph.graph['keywords'] = sorted(graph_freq_keywords.items(), key=lambda x: x[1], reverse=True)
        nxgraph.graph['embedding'] = self.generate_bpmn_embedding(nxgraph)

    def process_graph(self, bpmn_graph, text_attributes=['name', 'documentation'], min_text_length=1, max_keyphrase_length=1, threshold=0.5, logging=False):
        """
        Processes a BPMN graph, extracting keywords and embeddings for its nodes.

        Parameters:
        bpmn_graph (BPMNGraph): The graph to process.
        text_attributes (list of str): The attributes to use when extracting text from nodes.
        min_text_length (int): The minimum length of text to consider when extracting keywords.
        max_keyphrase_length (int): The maximum length of keyphrases to consider when extracting keywords.
        threshold (float): The minimum score for a keyword to be considered.
        logging (bool): Whether to log progress.

        Returns:
        BPMNGraph: The processed BPMN graph.
        """
        self.process_graph_nx(bpmn_graph.get_nx_graph(), text_attributes, min_text_length, max_keyphrase_length, threshold, logging)
        return bpmn_graph

class BPMNGraphEmbedderKeras(BPMNGraphEmbedder):
    """
    A class for embedding BPMN graphs using a Keras model.
    """
    def __init__(self, sentence_model='paraphrase-multilingual-MiniLM-L12-v2', graph_model='models/bpmn_search_0.1.0/bpmn_search_embedding_model.keras'):
        super().__init__(sentence_model)  # Call the parent class's constructor
        if not os.path.exists(graph_model):
            raise ValueError(f"Graph model {graph_model} does not exist.")
        logging.info(f"Loading keras model:{graph_model}")
        self.keras_model = load_model(graph_model, custom_objects=custom_objects)
        self.processor = GraphProcessor()

    def generate_bpmn_embedding(self, nxgraph):
        try:
            x = self.processor.get_node_features(nxgraph)
            x = np.expand_dims(x, axis=0)
            a = self.processor.get_adjacency_matrix(nxgraph)
            a = np.expand_dims(a.toarray(), axis=0)
            embedding = self.keras_model.predict([x, a], verbose=0)
            return embedding[0]
        except Exception as e:
            raise RuntimeError(f"An error occurred during the processing of the BPMN file: {e}")

    def process_graph_nx(self, nxgraph, text_attributes=['name', 'documentation'], min_text_length=1, max_keyphrase_length=1, threshold=0.5, logging=False):
        """
        Processes a NetworkX graph, extracting keywords and embeddings for its nodes.

        Parameters:
        nxgraph (networkx.Graph): The graph to process.
        text_attributes (list of str): The attributes to use when extracting text from nodes.
        min_text_length (int): The minimum length of text to consider when extracting keywords.
        max_keyphrase_length (int): The maximum length of keyphrases to consider when extracting keywords.
        threshold (float): The minimum score for a keyword to be considered.
        logging (bool): Whether to log progress.
        """
        graph_freq_keywords = self._extract_keywords_and_update_embeddings(nxgraph, text_attributes, min_text_length, max_keyphrase_length, threshold)
        nxgraph.graph['keywords'] = sorted(graph_freq_keywords.items(), key=lambda x: x[1], reverse=True)
        nxgraph.graph['embedding'] = self.generate_bpmn_embedding(nxgraph)
        nxgraph.graph['embedding_average'] = super().generate_bpmn_embedding(nxgraph)  # Use the parent class's method


    def process_graph(self, bpmn_graph, text_attributes=['name', 'documentation'], min_text_length=1, max_keyphrase_length=1, threshold=0.5, logging=False):
        """
        Processes a BPMN graph, extracting keywords and embeddings for its nodes.

        Parameters:
        bpmn_graph (BPMNGraph): The graph to process.
        text_attributes (list of str): The attributes to use when extracting text from nodes.
        min_text_length (int): The minimum length of text to consider when extracting keywords.
        max_keyphrase_length (int): The maximum length of keyphrases to consider when extracting keywords.
        threshold (float): The minimum score for a keyword to be considered.
        logging (bool): Whether to log progress.

        Returns:
        BPMNGraph: The processed BPMN graph.
        """
        self.process_graph_nx(bpmn_graph.get_nx_graph(), text_attributes, min_text_length, max_keyphrase_length, threshold, logging)
        return bpmn_graph


