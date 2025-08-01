from sentence_transformers import SentenceTransformer, util
import torch


class TaxonomyLabeler:
    def __init__(self, taxonomy_labels, model_name='all-mpnet-base-v2'):
        """
        Initialize the labeler with taxonomy labels and load the embedding model.
        """
        self.taxonomy_labels = taxonomy_labels
        self.model = SentenceTransformer(model_name)
        self.label_embeddings = self.model.encode(taxonomy_labels, convert_to_tensor=True, show_progress_bar=True)

    def label_errors(self, error_texts, threshold=0.6):
        """
        Assign taxonomy labels to a list of error texts.
        If threshold is provided, assign 'Other' if similarity below threshold.
        Returns:
            labels: list of label strings
            scores: list of highest similarity scores
        """
        error_embeddings = self.model.encode(error_texts, convert_to_tensor=True, show_progress_bar=True)
        cosine_scores = util.cos_sim(error_embeddings, self.label_embeddings)
        max_scores, best_indices = torch.max(cosine_scores, dim=1)

        labels = []
        for score, idx in zip(max_scores, best_indices):
            if threshold is not None and score < threshold:
                labels.append("Other")
            else:
                labels.append(self.taxonomy_labels[idx])

        return labels, max_scores.cpu().numpy()

