# NLP: Measuring Semantic Sentence Similarity Using Baseline and Neural Models

This project focuses on enhancing semantic sentence similarity by designing and implementing various Machine Learning and Deep Learning models in Natural Language Processing (NLP). By applying advanced techniques, the project aims to minimize the model's mean squared error and evaluate the semantic distance between words or sentences in vector space using metrics like cosine similarity and word mover's distance.

## Pre-Requisites

To get started with this project, you will need the following:

- **Python Environment**: Python 3 with an editor such as PyCharm or Anaconda.
- **Jupyter Notebook**: For running and experimenting with the code.
- **PyTorch**: Latest version (GPU not required).
- **Foundational Knowledge**: Basic understanding of Machine Learning, Deep Learning, and NLP.
- **Required Python Packages**:
  - `NLTK`
  - `sklearn`
  - `numpy`
  - `pandas`
  - `scipy`
  - `sent2vec`
- **Embeddings**:
  - BioSentVec embedding (required for `4BioSentvec.ipynb`): [Download from NCBI](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin).
  - Bio Word2Vec embedding (required for `3Word2Vec.ipynb`): [Download from Bio NLP Lab](https://bio.nlplab.org).
- **Dataset**:
  - Request access from [Harvard DBMI Portal](https://portal.dbmi.hms.harvard.edu).
  - Task 2 dataset available at [OHNLP 2018](https://sites.google.com/view/ohnlp2018/home).

## Research Papers

The following research papers provide valuable context and insights for this project:

1. Wang Y, Afzal N, Liu S, Rastegar-Mojarad M, Wang L, Shen F, Fu S, Liu H. **Overview of the BioCreative/OHNLP Challenge 2018 Task 2: Clinical Semantic Textual Similarity.** Proceedings of the BioCreative/OHNLP Challenge, 2018.
2. Wang Y, Afzal N, Fu S, Wang L, Shen F, Rastegar-Mojarad M, Liu H. **MedSTS: A Resource for Clinical Semantic Textual Similarity.** Language Resources and Evaluation, 2018.
3. Chen Q, Peng Y, Lu Z. **BioSentVec: Creating Sentence Embeddings for Biomedical Texts.** Proceedings of the 7th IEEE International Conference on Healthcare Informatics, 2019.

## Deployment

- **Docker Image**: Available at `/aswaths/semantic_similarity:v3`
- **Streamlit Application**: [Semantic Similarity Streamlit App](https://semanticsimilarity.streamlit.app)
- **Inference API**: Hosted on Render - [Semantic Similarity API](https://semantic-similarity-v3.onrender.com/docs)

