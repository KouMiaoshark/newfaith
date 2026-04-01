## Data

Dataset for fine-tuning Seq2seq models including intermediate questions generated via GPT, and annotated tsf data via distant supervision. 
### timequestions
- intermediate_questions
  - We use in-context learning to obtain the data set. 
  - Number of data
    - train: 847
    - dev: 287
  - Each question mainly includes the following contents.
     - "Id": question id in TimeQuestions
     - "Question": question in TimeQuestions
     - "Temporal signal": Temporal signals
     - "Temporal question type": Temporal question type inlcuding Explicit, Implicit, Ordinal, and Temp.Ans 
     - **"silver_generated_question": generated intermediate question and answer type from GPT**
     - "Data source": data source of TimeQuestions
     - "Answer": ground truth answer including Wikidata Qid, Wikidata label, and Wikipedia URL
- tsf_annotation
  - We apply distant supervision to annotate the dataset 
  - Number of data
    - train: 9,708
    - dev: 3,236

### tiq
- intermediate_questions
  - Number of data
    - train: 5,875
    - dev: 1,949
  - Each question mainly includes the following contents.
     - "Id": question id in TIQ
     - "Question": question in TIQ
     - "Temporal signal": Temporal signals such as OVERLAP, AFTER, and BEFORE
     - "Temporal question type": In TIQ, all the types are "Implicit"
     - **"silver_generated_question": generated question and answer type from GPT**
     - "Data source": TIQ
     - "Answer": ground truth answer including Wikidata Qid, Wikidata label, and Wikipedia URL

- tsf_annotation
    - We apply distant supervision to annotate the dataset and combine TIQ with temporal value questions.
    - Number of data
      - train: 13,723 (6,000 (tiq) + 7,723(temporal value questions))
      - dev: 4,542 (2,000 (tiq) + 2,542 (temporal value questions))

