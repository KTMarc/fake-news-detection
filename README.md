# Fake News Detection with Recurrent Neural Networks
Final project for the UPC postgraduate in Deep learning and Aritificial Intelligence

**Postgraduate course: Artificial Intelligence with Deep Learning (UPC School, 2026)**

| | |
|---|---|
| **Team members** | Valentina Martínez · Selena Rodríguez · Marc Humet |
| **Advisor** | Pol Caselles |
| **Dataset ISOT** | [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php) (~44K news articles) |
| **Dataset LIAR** | [LIAR Dataset](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset) (~42K news articles) |
| **Task** | Binary classification: Fake (0) vs True (1) |


## How to Run

### Prerequisites

- Google account (for Colab + Drive)
- Copy the folder `aidl-final-project` in the root of your Google drive

### 1. Preprocessing

Open `ISOT_Preprocessing.ipynb` in Google Colab. It will:
- Request access to google drive. You should grant it in order to see the Datasets and write the results there
- Remove duplicates, split into train/val/test
- Clean and tokenize train/val (test left uncleaned)
- Build vocabulary, convert to padded ID tensors
- Save artifacts to `Google Drive/aidl-final-project/artifacts/`

### 2. Training

Open `ISOT_Training.ipynb` in Google Colab:
1. Set `MODEL_TYPE` in the Configuration cell (`"RNN"`, `"GRU"`, or `"LSTM"`)
2. Adjust hyperparameters as needed
3. Run all cells

Results are saved to `Google Drive/aidl-final-project/outputs/{model_type}/`.


1. ## **Introduction**

The primary goal of this project is to develop and evaluate a supervised machine learning model capable of autonomously classifying news articles as either True or Fake. By leveraging Natural Language Processing (NLP) techniques, we aim to identify linguistic patterns and statistical regularities that distinguish factual reporting from misinformation.

## **Motivation**

The motivation for this research is driven by the urgent need to preserve information integrity in an era where the rapid dissemination of misinformation poses a direct threat to public opinion and social stability. Given the overwhelming volume of data generated daily, manual fact-checking has become insufficient, making automated and scalable solutions essential. By leveraging Artificial Intelligence, this project aims to transcend human cognitive biases through advanced pattern recognition. Unlike traditional human analysis, AI can identify  linguistic structures and statistical regularities characteristic of deceptive content, providing a robust, data-driven defense against digital disinformation.

## **Proposal**

This project utilizes three different many-to-one recurrent architectures—RNN, LSTM, and GRU—to classify news articles from the public ISOT dataset. The main objectives are:

* Test an end-to-end pipeline for the GRU architecture, encompassing preprocessing, embedding, model training, and testing.  
* Analyze the textual features (such as vocabulary, style, and structure) that the models leverage to distinguish between fake and real news.  
* Compare the performance of the RNN, LSTM, and GRU models using standard metrics: precision, recall, and F1-score

2. ## **Dataset: ISOT Fake News Dataset**

The **ISOT Fake News Dataset** is a benchmark dataset for binary text classification. It comprises over 44,000 news articles compiled between 2016 and 2017\. The data is divided into two primary categories:

* **Real News (True Class):** Articles sourced from *Reuters.com*, ensuring a standard of neutrality and veracity.  
* **Fake News (Fake Class):** Articles from websites flagged as unreliable by fact-checking organizations such as PolitiFact and Wikipedia.

The distribution of the classes is as follows:

| Class | Amount of articles | Percentage |
| :---- | :---- | :---- |
| True  | 21,417 | 47.7% |
| Fake  | 23,481 | 52.3% |
| **Total** | **44,898** | **100%** |

For this project, the data was shuffled and split into training, validation, and test sets. A standard 60/20/20 split was applied to ensure the model can be evaluated on unseen data while maintaining enough samples for a robust training phase.

---

## **2.1 Dataset Interpretation**

Exploratory Data Analysis (EDA) revealed significant structural and lexical differences between the two classes:

* **Article Length:** Real news follows a bimodal distribution, containing both short reports and long-form articles. In contrast, fake news tends to follow a unimodal distribution with more uniform lengths.  
* **Sesgo Temático y Data Leakage:** Se detectó que la columna subject (tema) y la presencia recurrente del tag "Reuters" en las noticias reales podrían causar "filtración de datos" (*Data Leakage*). Si el modelo aprende que "Reuters \= Verdad", no estará detectando desinformación, sino identificando una marca de agua. Por ello, se decidió eliminar referencias directas a la fuente durante la limpieza.  
* **Riqueza Léxica:** El vocabulario sigue la **Ley de Zipf**, donde un grupo reducido de palabras clave (política, gobierno, Trump) domina la frecuencia, mientras que existe una "cola larga" de términos específicos que aparecen una sola vez (*hapax legomena*).

---

## **2.2 Transformaciones aplicadas (Preprocessing)**

Para transformar el texto bruto en un formato que la red neuronal pueda procesar, se implementó un *pipeline* de preprocesamiento personalizado en Python:

1. **Concatenación:** Se fusionaron las columnas title y text en una única variable para capturar la relación semántica entre el titular y el cuerpo de la noticia.  
2. **Limpieza de Ruido:** \* Conversión a minúsculas.  
   * Eliminación de caracteres especiales y números mediante expresiones regulares.  
   * Filtrado de *stopwords* (palabras frecuentes sin carga semántica como "the", "is", "at").  
3. **Tokenización y Padding:** El texto se dividió en unidades mínimas (tokens) y se aplicó *padding* para que todas las secuencias de entrada tengan una longitud fija, facilitando el procesamiento en lotes (*batches*).  
4. **Normalización:** Se filtraron palabras con una longitud inferior a 3 caracteres para reducir el espacio de características y eliminar errores tipográficos o ruido innecesario.

## Training

The model had good training results from the very beginning, when we started with the GRU model with the first hyperparameters we tried:

Embedding dimensions: 128  
Hidden size: 128  
Batch size: 32  
LR: 0,001  
GRU Layers: 1  
Bidirectional

Because of that, we didn´t have to fine tune the initial hyperparameters and we didn't consider using word2vec or Glove to transfer learning for those models. The random initialization of embeddings was good enough.  
                
Preprocessing and training were all in the same Google colab notebook. At some point in the project we decided to split it in two. So training and preprocessing could evolve independently.

Because the model worked well, we focused on the preprocessing part and finding what was potentially biasing the model was our goal. 

We introduced RNN and LSTM and we still had some good results, so we decided that a transformer was not needed and an overkill for the task. At least while we were not sure about our dataset was the problem.

# 3\. Experiments

This section presents the experiments conducted to better understand why the model achieved very high performance on the ISOT dataset. The main goal was not only to optimize the classifier, but also to investigate whether the results were influenced by dataset-specific artifacts, lexical shortcuts, or duplicated samples. Several experiments were therefore designed to test the robustness of the GRU model under different preprocessing settings, model architectures, and datasets.

## 3.1 Cleaning the ISOT Dataset

**Hypothesis**   
The dataset may contain elements that make the classification task artificially easy, allowing the model to achieve unusually high performance.

**Experiment Setup**

Additional cleaning steps were applied to the dataset:

* normalization was used to reduce lexical variation by grouping similar forms of words;  
* very infrequent words were filtered out;  
* duplicate samples were identified and removed.  
* Test the new input with the GRU Model

**Results**

* The duplicate analysis showed that approximately **25% of the dataset consisted of duplicate entries**.   
* The vocab size was reduced around a **50%** (Original size \~96,654 and it was reduced to \~46,864)  
* The result of loss, accuracy in the training and the metrics in the test, only variant in a small portion (0.01 \~ 0.001)

![][image1]  
Image Z: Frequency of the word appearance distribution

![][image2]  
Image Z+1: Training loss and accuracy of ISOT before and after clean

(IMAGE of the HISTOGRAM OF FREQ VS WORDS (before and after))

**Conclusion**:  
Cleaning the dataset reduced the vocabulary size and made the corpus more consistent. This preprocessing step also made training the GRU model more efficient. The hypothesis was not true, because the GRU model had very similar results.

## 3.2 Excluding Exclusive Words from Fake and Real News

### Hypothesis

The model may be relying on words that are highly specific to one class rather than learning generalizable patterns of misinformation detection.

Topic-specific words that appear almost exclusively in true articles — such as myanmar, rohingya, hariri— could act as spurious shortcuts. If the model relies on them, masking these words at test time should cause a measurable drop in performance.

### Experiment Setup

Rather than retraining the model with different word lists (which introduces variance from random initialization), we:

1. Train the model once  
2. Evaluate the frozen model on the test set with the top-N true leaky words replaced by \<pad\>, for N in 0-15

The leaky words are sourced from words with the highest true/fake ratio (e.g., myanmar appears in 3,084 true articles but only 4 fake ones).  
![Leaky words](Images/leaky_words.png)

### Results

The analysis revealed that: 

* **“Reuters”** appeared in a large proportion of true news articles. But was already removed during preprocessing in train/val.  
* Several words occurred thousands of times almost exclusively within their respective classes. This confirmed the presence of strong lexical markers that could bias the model. 

(Add histogram of fake and real words vs freq before and after the changes)

![GRU leaky words](Images/gru_leakywords.png)

### Conclusion

Even after cleaning the data, the GRU model continued to be robust to leaky word removal. This suggests that, although some obvious lexical shortcuts were identified, the dataset may still contain residual class-specific artifacts that make the task easier than real-world fake news detection.

So it was decided to try a new dataset.

## 3.3 GRU training with different datasets 

**Hypothesis**  
The almost perfect scores obtained on ISOT may be due to the model learning dataset-specific patterns. If this is the case, then performance should decrease when the same GRU model is trained on a different dataset or on a more diverse combined dataset. 

**Experiment Setup**

To test this hypothesis, the same GRU architecture and training configuration were applied to different datasets. In addition to ISOT, the **LIAR dataset** was incorporated based on recommendations from previous literature. The experiments considered three training conditions:

* **ISOT** only,  
* **LIAR** only,  
* **BOTH**, a combined dataset including ISOT and LIAR. 

The GRU model was trained using the following hyperparameters:

* embedding dimension: **128**  
* learning rate: **0.001**  
* batch size: **32**  
* number of epochs: **3**  
* early stopping patience: **3**  
* hidden size: **64**  
* bidirectional GRU: **True**. 

The LIAR dataset was simplified into two categories, **real** and **fake**, resulting in **12,791 samples**:

* Real: **7,134** samples (**55.8%**)  
* Fake: **5,657** samples (**45.2%**). 

**Results**  
In the following graphic you will see the loss and accuracy using the three dataset:  
![][image5]  
Image N: Loss and validation graphic of the GRU training with different dataset

The test results obtained with the trained GRU model were as follows:

| Dataset | Class | Precision | Recall | f1-Score |
| :---- | :---- | ----- | ----- | ----- |
| ISOT | True | 0.98 | 1.00 | 0.99 |
|  | Fake | 0.99 | 0.96 | 0.98 |
| LIAR | True | 0.63 | 0.43 | 0.51 |
|  | Fake | 0.48 | 0.67 | 0.56 |
| BOTH | True | 0.92 | 0.74 | 0.82 |
|  | Fake | 0.69 | 0.90 | 0.78 |

**Conclusion**

The ISOT-based model achieved extremely high results, with near-perfect precision, recall, and F1-scores on the test set. Although these metrics are strong, they should be interpreted with caution. In a task as complex as fake news detection, performance that is almost perfect is unusual.

By contrast, the LIAR dataset produced much lower scores, suggesting that it is a more difficult and more realistic benchmark. Its shorter statements, higher variability, and less regular structure make it harder for the model to rely on superficial cues. The combined dataset produced intermediate results, which further supports the idea that greater diversity reduces artificial performance gains. 

Overall, these experiments show that model performance depends heavily on the dataset used. The results suggest that the strong performance observed on ISOT is influenced not only by the architecture itself, but also by the specific properties of that dataset. 

## 3.4 Comparing the different Model with ISOT\&LIAR dataset

**Hypothesis**:  
GRU is expected to perform strongly because of its bidirectional structure and efficient handling of sequential text data; it may not consistently outperform LSTM and RNN across all datasets. Therefore, this experiment aims to compare the three architectures and identify which model generalizes better under different data conditions.

**Experiments Setup**

To evaluate the impact of model architecture, three recurrent neural network variants were tested: **RNN, LSTM, and GRU**. Each model was trained and evaluated on three dataset configurations:

* **ISOT only**  
* **LIAR only**  
* **BOTH**, a combined dataset including ISOT and LIAR

All models were trained under the same experimental conditions to ensure a fair comparison. The hyperparameters used were:

* embedding dimension: 128  
* learning rate: 0.001  
* batch size: 32  
* number of epochs: 10  
* early stopping patience: 3  
* hidden size: 64  
* bidirectional: True  
* number of RNN layers: 1  
* dropout (RNN): 0.4

**Results**

ISOT Dataset Performance Comparison

On the **ISOT dataset**, all three models achieved strong performance, but **LSTM obtained the best overall results**, with the highest precision, recall, and F1-scores across both classes. This suggests that ISOT is a relatively easy dataset for recurrent architectures and may contain patterns that are easy to learn.

| Model | Fake Precision | True Precision | Fake Recall | True Recall | Fake F1 | True F1 |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- |
| RNN | 0.98 | 0.93 | 0.87 | 0.99 | 0.92 | 0.96 |
| GRU | 0.96 | 0.91 | 0.84 | 0.98 | 0.90 | 0.95 |
| LSTM | 0.99 | 0.95 | 0.91 | 1.00 | 0.95 | 0.97 |

![][image6]  
Image Y: ISOT dataset model with RNN, LSTM and GRU

BOTH Dataset Performance Comparison 

On the **combined dataset (BOTH)**, performance decreased compared with ISOT, indicating that the inclusion of LIAR increases task difficulty and dataset diversity. In this setting, **GRU produced the most balanced results**, achieving stronger overall performance than RNN and more consistent class-level behavior than LSTM.

| Model | Fake Precision | True Precision | Fake Recall | True Recall | Fake F1 | True F1 |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- |
| RNN | 0.77 | 0.76 | 0.56 | 0.89 | 0.65 | 0.82 |
| GRU | 0.78 | 0.85 | 0.77 | 0.86 | 0.78 | 0.86 |
| LSTM | 0.70 | 0.91 | 0.89 | 0.75 | 0.78 | 0.82 |

![][image7]  
Image Y: BOTH dataset model with RNN, LSTM and GRU

LIAR Dataset Performance Comparison 

On the **LIAR dataset**, all models performed considerably worse than on ISOT. This confirms that LIAR is a more challenging and realistic benchmark. The differences between models were smaller, and no architecture clearly dominated in all metrics. However, **GRU and LSTM showed slightly stronger results than RNN**, depending on the class and metric considered.

| Model | Fake Precision | True Precision | Fake Recall | True Recall | Fake F1 | True F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RNN | 0.47 | 0.61 | 0.67 | 0.41 | 0.55 | 0.49 |
| GRU | 0.50 | 0.63 | 0.58 | 0.56 | 0.54 | 0.59 |
| LSTM | 0.51 | 0.60 | 0.43 | 0.67 | 0.46 | 0.64 |

**![][image8]**

Image Y: LIAR dataset model with RNN, LSTM and GRU

**Conclusion**

These experiments confirm that **dataset characteristics have a greater impact on performance than model choice alone**. While recurrent architectures perform very well on ISOT, their results drop significantly on LIAR and remain intermediate on the combined dataset. This suggests that the near-perfect performance observed on ISOT is likely influenced by dataset-specific patterns rather than true robustness in fake news detection.

Therefore, the hypothesis that **GRU is the best overall model is only partially supported**. GRU performs well, especially on the combined dataset, but **LSTM achieves the best results on ISOT**, and no single model is consistently superior across all datasets. At this stage, improving **data quality, diversity, and balance** appears to be more important than moving to a more sophisticated architecture.

# 4\. Next Steps

* Increase the source of the dataset to balance the amount of ISOT dataset to evaluate the performance of the model based on the hyperparameters or the model architecture.  
* Tune the different models using the LIAR and the combination with ISOT dataset to evaluate the performance.

# Bibliography

* Fake News Detection: Comparative Evaluation of BERT-like Models and Large Language Models with Generative AI-Annotated Data [https://arxiv.org/pdf/2412.14276](https://arxiv.org/pdf/2412.14276)  
* Localization of Fake News Detection via Multitask Transfer Learning [https://arxiv.org/pdf/1910.09295](https://arxiv.org/pdf/1910.09295)
