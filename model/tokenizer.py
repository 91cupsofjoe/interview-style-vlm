from model.model import Transformer


class Tokenizer(Transformer):
    """
    Class for the word tokenizer.
    """
    def __init__(self,
        preset_tokens_ids=None
    ):
        # If no preset mapping of tokens to ids is provided, initialize it
        if preset_tokens_ids is None:
            preset_tokens_ids = {}

        # Initialize the mapping of tokens to ids
        self.token_ids = preset_tokens_ids

    def get_token_id(self, token):
        """
        Return the token id

        Args:
            token (str): The token

        Return:
            The id of the token
        """
        # Return the word ID for the token ('<PAD>' for an unknown token)
        if token not in self.token_ids:
            return self.token_ids['<PAD>']
        return self.token_ids[token]

    def tokenize_sentences(self, sentences: list[str]):
        """
        Parse a list of sentences into tokens and store the token ids.

        Args:
            sentences (list): The list of sentences

        Return:
            None
        """
        # Get the set of unique tokens across all sentences
        token_set = set([])

        # Iterate through the sentences
        for sentence in sentences:
            # Iterate through the tokens of the sentence
            for token in sentence:
                token_set.add(token)

        # Add an entry for every unique token encountered
        for token in token_set:
            self.token_ids[token] = len(self.token_ids)

    def tokenize_sentence(self, sentence):
        """
        Parse a sentence into tokens and store the token ids.
            Single-sentence variant of tokenize_sentences.
        """
        self.tokenize_sentences([sentence])