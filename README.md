# High Recall Information Retrieval System Applied to Social Science Research

![Contributors](https://img.shields.io/github/contributors/mmaisonnave/unplanned-hospital-readmission-prediction?style=plastic)
![Forks](https://img.shields.io/github/forks/mmaisonnave/unplanned-hospital-readmission-prediction)
![Stars](https://img.shields.io/github/stars/mmaisonnave/unplanned-hospital-readmission-prediction)
![GitHub](https://img.shields.io/github/license/mmaisonnave/unplanned-hospital-readmission-prediction?style=round-square)
![Issues](https://img.shields.io/github/issues/mmaisonnave/unplanned-hospital-readmission-prediction)


## High Recall Information Retrieval System Overview:
One of the key features of the repository is its Jupyter Notebook-based graphical user interfaces for performing high-recall information retrieval. The goal is to assist users in identifying all relevant documents from a large pool with minimal annotations. The system operates iteratively, suggesting batches of documents to label. It employs a smart selection strategy to identify documents that will maximize the classifier's learning.

For each suggestion, the user labels documents as either "relevant" or "irrelevant" based on their information needs. The system relies on one or more machine learning models to classify each document as "relevant" or "irrelevant." Once the annotation process is complete, the system applies the final ML models to all the documents and exports those classified as relevant. The repository features two versions: one that uses active learning and another that incorporates Scalable Continuous Active Learning (SCAL).

For example, in one of our research studies, we utilized the SCAL system to analyze 2,961,906 articles, labeling just 315 of them, and filtered the results down to 17,014 relevant articles.


## High Recall Information Retrieval using Active Learning (First version)
The main files of the first version are `utils.ui.py` and `utils.high_recall_information_retrieval.py`. The notebook `System V1.0.ipynb` is used to launch the first version of the system with the following piece of code:

```Python
from utils.ui import UI

ui = UI()
ui.run()
``` 

The user interface class (`utils.ui.UI`) has a `system` attribute of type `utils.high_recall_information_retrieval.HRSystem` which handles all the logic of the high recall information system. The user interface allows the user to interact with the system. The possible actions are:

```Python
    high_recall_information_retrieval.HRSystem.loop(...)
```
The `loop` method is part of an active learning process where a model suggests data points that need to be labeled by a user. It starts by preparing and checking the data to ensure no interruptions or duplicates, then calculates new suggestions based on the model’s current understanding. These suggestions are presented to the user for annotation, and once labeled, they are added to the model’s training data. The model is then retrained with the newly labeled data. This process repeats iteratively, improving the model each time by incorporating new, labeled examples. If no new suggestions are found, the `loop` is skipped, but a final function is always called to handle any cleanup. The method handles the calling of the `after_labeling` and the cancel and finish functions (`cancel_process_fn`, `finish_process_fn`).

```Python
    high_recall_information_retrieval.HRSystem.save()
```
The `save` method saves the current state of the system, including labeled data, unlabeled data, and iteration count, to disk. The method is called when the user clicks the `SAVE` button on the user interface.


```Python
    high_recall_information_retrieval.HRSystem.export(...)
```

The `export` method exports relevant articles from the labeled and potentially suggested data to a CSV file and optionally sends the file via email. The suggestions are calculated with a ML model trained with the latest data. 


```Python
    high_recall_information_retrieval.HRSystem.review_labeled(...)
```

The method `review_label` allows the user to correct already labeled items. It allows reviewing a specified number of labeled items, allowing for corrections to their labels. After reviewing, it retrains the model if changes are made to the labels. The method displays the reviewed items, compares the original and new labels, and logs the results.


```Python
    high_recall_information_retrieval.HRSystem.status()
```

The `status` method displays the status of the HRSystem, including details about labeled and unlabeled data, relevant and irrelevant articles, classifier performance metrics, and confusion matrix for each model in the system.

```Python
    high_recall_information_retrieval.HRSystem.get_labeled_count()
    high_recall_information_retrieval.HRSystem.get_unlabeled_count()
    high_recall_information_retrieval.HRSystem.get_relevant_count()
```

The methods `get_labeled_count`, `get_unlabeled_count`, and `get_relevant_count` return the number of labeled instances, the number of unlabeled instances, and the number of relevant items in the labeled set, respectively.

## High Recall Information Retrieval using SCAL (Last versions)
After the first version, I updated the implementation of high-recall information retrieval to utilize the Scalable Continuous Learning (SCAL) approach. The following Jupyter notebooks employ the SCAL methodology:

1. System.ipynb
2. System_2nd_round.ipynb
3. SCAL_system.ipynb
4. SCAL_system_second_round.ipynb

As part of our research project, we conducted a two-step SCAL-based information retrieval process for two different topics:
- Displaced persons (DP)
- Multiculturalism (MC)

For the first topic (DP), steps one and two were carried out using notebooks (1) and (2), respectively. For the second topic (MC), steps one and two were executed using notebooks (3) and (4), respectively.

All four SCAL-based notebooks use the same main Python modules:

```Python
from utils.data_item import DataItem
from utils.oracle import Oracle
from utils.scal_ui import SCAL_UI
from utils.scal import SCAL
from utils.data_item import QueryDataItem

```

In the remainder of this section, I give an overview of the main functionalities provided by those modules.

**`utils.data_item.DataItem`**
Class used to represent real and synthetic data items with functionality for label management, vector generation, and metadata extraction.


**`utils.oracle.Oracle`**
The Oracle class offers methods for managing and analyzing a collection of data items, with a specific focus on relevance labeling and generating document representations. It enables the retrieval and handling of a pre-labeled collection of items, serving as a ground-truth oracle by allowing users to query the relevance label for any instance in the collection. The collection included in the oracle was created through manual annotation.


**`from utils.data_item.QueryDataItem`**
The `QueryDataItem` class represents a synthetic data item specifically designed for query-based analysis. It enables the conversion of a user input query into a data item, providing an interface consistent with the `utils.data_item.DataItem` class.


**`utils.scal_ui.SCAL_UI`** and **`utils.scal.SCAL`**
The notebooks create a user interface using the `utils.scal_ui.SCAL_UI` class. It obtains the session name and topic description through this interface and uses them to launch a SCAL system by invoking the `utils.scal.SCAL` class. The SCAL system uses the `myversions.pigeonXT` class to create another user interface to request annotations from the user and manage the entire user interaction thereafter.


The `utils.scal_ui.SCAL_UI` user interface obtains session name and topic and creates (or loads from disk) an SCAL system using the method `utils.scal.SCAL.run`.

```Python
    # The notebook, on a high-level creates the method to create a new system, creates a user interface
    # to request a session_name and a topic_description, and it use those two to create a SCAL system and
    # call run().
    # The simplified notebook looks like:
    def start_system(session_name, topic_description)
        # Either loads or creates a SCAL system and invokes run.
        # ...
        # The creation of a new systems looks like the following:
        scal = SCAL(session_name, 
                    labeled, 
                    unlabeled,
                    random_sample_size=N,
                    batch_size_cap=cap,
                    simulation=False, 
                    seed=seed)
        scal.run()

    _=SCAL_UI(start_system, second_round=True)
```

After the system is running, the `run` method repeatedly calls `utils.scal.SCAL.loop` using a structure that consists only of:
```Python
    def run(self):
        if self.j<self.cant_iterations: # Iteration number didn't exceed the maximum number of iterations
            self.loop()
    # end of run
```

The `utils.scal.SCAL.loop` method orchestrates the iterative process of labeling and model training in the SCAL system. It begins by extending the labeled dataset with randomly selected documents and trains a new classification model. Using the updated model, it identifies the most relevant documents from the unlabeled pool and samples a batch for further annotation. These documents are presented to the user for labeling (unless in simulation mode, where the process advances automatically and the labels are obtained from the `Oracle`). The method updates annotations and tracks progress, iteratively refining the classifier and selecting documents until the `loop` completes all iterations. It ensures model storage, computes recall thresholds, and prepares the system for the final classification step. The `loop` method invokes the `myversions.pigeonXT.annotate` method from the `pigeonXT` class, which prompts the user for labels through a minimalist user interface.


The two main methods, besides `loop`, in the SCAL system are:**`utils.scal.SCAL.after_loop`** and **`utils.scal.SCAL.finish`**. 


The `after_loop` method manages updates and maintain the system's state after each iteration of the SCAL process. It tracks labeled and unlabeled items, updates user-provided or simulated labels, and ensures consistency in the labeling. The method recalculates key metrics like precision, true positives, and recall estimates. Additionally, it expands the labeled collection, removes processed items from the unlabeled pool, and logs progress for debugging and evaluation. If there are unlabeled items remaining, it prepares for the next iteration; otherwise, it finalizes the process.

The `finish` method completes the SCAL process by finalizing the labeling of data. It calculates important metrics like the expected number of relevant articles and sets a threshold for classifying data as relevant. For this last classifier, we use the threshold computed using the SCAL methodology to ensure that we achieve the desired level of recall. It then saves the labeled data to a file, builds a final model to improve predictions, and applies this model to the remaining unlabeled data to identify potentially relevant items. If the process is running in simulation mode, it also evaluates the model's accuracy, precision, recall, and other performance metrics. 


**Note**: The first versions of the high-recall information retrieval system will not work due to substantial changes made to Python modules during the implementation of the SCAL approach. For example, the `myversions.pigeonXT` is used differently in the first and later versions of the system. To run previous versions, the entire repository can be cloned from early commits to access and run the old versions.


## Database Used for the two research topics

For the 'displaced persons' research topic, we used data from The Globe and Mail newspaper, covering the period from 1945 to 1967. Initially, 6,938 articles mentioning 'Canada' and 'DP' (displaced persons) were collected through a manual search assisted by a search engine. From these, 2,038 articles were manually reviewed, and 522 were identified as relevant (about 25.6%). The corpus was then expanded to 2,057,868 articles from the same period, enabling the researchers to apply machine learning models for more efficient identification of relevant articles. To analyze the 2+ million articles, we experimented with various versions of high-recall information retrieval systems, ultimately adopting SCAL (Scalable Continuous Active Learning), which was used from the end of this research topic through the entirety of the next.

For our research on multiculturalism, we analyzed nearly three million articles from *The Globe and Mail* newspaper, covering the period from 1960 to 2018. Using the SCAL high-recall information retrieval approach, we identified over 8,000 relevant documents while labeling only a few hundred articles. Through topic modeling and manual topic analysis, we further narrowed the number of relevant articles to just over 2,800.

The data and computing resources were obtained through the TDM Studio platform. 


## Other Exploratory Scripts and functionality:
Throughout the research, we explored various modifications and alternatives to improve the system’s performance and robustness. This included experimenting with different ranking functions to decide which documents should be shown to the user. For instance, we tested functions that prioritized suggestions based on the classifier's uncertainty, or those that highlighted documents the classifier identified as relevant with high confidence.

To evaluate the effectiveness of the information retrieval process, we conducted topic analysis and applied clustering techniques to the results. For instance, in the notebooks/analysis_of_8k_suggestions/ folder, we applied DBSCAN and Top2Vec clustering algorithms to analyze the 8,000 articles resulting from the two rounds of SCAL applied to the original corpus of nearly three million articles.

We also used the 20 Newsgroups dataset to simulate various high-recall information retrieval system variants. In this experiment, one topic was labeled as relevant, and the other 19 were considered irrelevant, enabling us to assess the performance of the system under controlled conditions. Similarly, we simulated the performance of the system on the manually-annotated part of the displaced persons dataset.

For more details on the simulation results using these datasets, refer to the repositories:

- [Link to repository 1](https://github.com/mmaisonnave/hrir-train-test-simulations)
- [Link to repository 2](https://github.com/mmaisonnave/hrir-simulation-results)

Additionally, in the notebooks/10_computing_new_scores_on_old_suggestions_(17k_and_8k) folder, we analyzed around 17,000 suggestions from the first round of SCAL and 8,000 suggestions from the second round for the multiculturalism research topic.

"Once the suggestions were generated for the multiculturalism topic, we evaluated their quality by sorting them based on the classifier’s confidence. We then randomly selected examples from different parts of the ranked list to understand how performance declined towards the end, where suggestions are more likely to be irrelevant. A similar evaluation process was conducted for the displaced persons dataset, which is detailed in the Jupyter notebook `notebooks/dp_real_world_excercise/Creating_evaluation_of_second_round_50_labels.ipynb.`"

## Dependencies
We ran all experiments using the dependencies and libraries listed in the `requirements_imm.txt` file. We used two additional Python virtual environments, one for calculating top2vec (`requirements_top2vec.txt`) and another to compute sentence bert embeddings (`requirements_sbert_sample_env.txt`).

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## Contributing
We welcome contributions from the community. Please open an issue or submit a pull request for any improvements or suggestions.

## Acknowledgments
We gratefully acknowledge the financial support of a 2-year New Frontiers in Research Fund (NFRF) (grant # NFRFE-2020-00996).
