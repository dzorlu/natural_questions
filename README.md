(WIP)

# Natural Questions
Google's [Natural Questions corpus](https://ai.google/research/pubs/pub47761),
is a question answering dataset in which questions consist of queries issued to the
search engine whereas answers are corresponding Wikipedia pages. The answers are annotated in long and short form.
The long and short annotations might be empty however - The answer to the search query can be a Yes / No answer or
the Wikipedia entry might not contain the answer. The breakdown is best illustrated in the original paper reproduced below.



![Figure 1](https://github.com/dzorlu/natural_questions/blob/master/supporting_docs/Figure%201.png)


This repo replicates the code and results presented in [A BERT Baseline for the Natural Questions](https://arxiv.org/abs/1901.08634).

## Pre-processing
Preprocessing script generates all possible instances of the context for a given training example with a sliding window approach,
appended to the question tokens and special characters. Because the instances for which an answer is
present is very sparse, the script downsamples all null instances with a ratio of 500:1 in order to generate a balanced dataset.

If a short answer is present within a span, the indices point to the smllest span containing all annotated short answer spans.
If only a long answer is present, indices point to the span of the long answer. If neither is found, indices
point to the "[CLS]" token.

The development data contains 5-way annotations on 7,830 items. If at least 2 out of 5 annotators have given a non-null answer on the
example, then the system is required to output a non-null answer that matches one of the 5 annotations;
conversely if fewer than 2 annotators give a non-null long answer, the system is required to return NULL as its output.

Hence, processing the evaluation dataset, the module discards any span that has less than two target annotations. The annotations, in turn,
are used at eval time to estimate the model loss, precision, and recall on span basis. Measuring the accuracy of the model both on span basis
and the document basis allowed me to work on modeling and post-processing components separately.


```buildoutcfg
python preprocessing/preprocessing.py \
    --bert_data_dir $BERT_DATA_DIR \
    --max_seq_length 384 \
    --max_query_length 64 \
    --doc_stride 128
```


## fine-tuning
Fine tuning extends the [BERT library](https://github.com/google-research/bert).
A single layer `W` on top that transforms BERT outputs onto span predictions. The entire model is fine tuned jointly to
maximize the log-probability of the correct span.

```
 python run_nq.py \
    --bert_config_file $CHECKPOINT_DIR/bert_config.json  \
    --vocab_file  $CHECKPOINT_DIR/vocab.txt \
    --output_dir $OUTPUT_DIR \
    --bert_data_dir $BERT_DATA_DIR \
    --num_train_steps 100000 \
    --do_train True \
    --max_seq_length 384 \
    --init_checkpoint $CHECKPOINT_DIR/bert_model.ckpt \
    --train_batch_size 8
    --learning_rate 0.00005 \
    --save_checkpoints_steps 1000
```

The model is trained approximately for 3 epochs. The span predictions are calculated taking the maximum combined score
of start and end token in a given span, where end token index must be larger or equal to start token.

## inference / post-processing
Postprocessing step aggregates span predictions to the document level. In particular, if no answer is present in any of the spans for a document,
the postprocessing script assigns a null-answer to the document. if an answer is present for a given document, the script takes the answer
with the highest score. Unlike the baseline model, that emits a prediction for each document and then adjusts the answers based on the score distribution,
I leave the document-level predictions same. A well-calibrated model should match the the target distribution without any pruning or adjustments.


```


```

## error analysis

### V1
A detailed error analysis can be found in the [notebook](https://github.com/dzorlu/natural_questions/blob/master/error_analysis.ipynb).

1 - There are many examples where the model prediction looks valid, but the development annotations fail to match the prediction span.

```
{'gold_answers': ["Destiny 's Child",
  'Solange Knowles',
  "Destiny 's Child",
  'Solange Knowles'],
 'gold_span': [(187, 189), (184, 185), (187, 189), (184, 185)],
 'model_answers': "Solange Knowles & Destiny 's Child",
 'model_span': (184, 189),
 'question_text': 'who sings the theme song for the proud family',
 'url': 'https://en.wikipedia.org//w/index.php?title=The_Proud_Family_(soundtrack)&amp;oldid=638462881'}
```

```
{'gold_answers': ['14', '14', '14', '14'],
 'gold_span': [(68, 68), (235, 235), (235, 235), (235, 235)],
 'model_answers': '14 member countries',
 'model_span': (944, 946),
 'question_text': 'how many countries are a part of opec',
 'url': 'https://en.wikipedia.org//w/index.php?title=OPEC&amp;oldid=834259589'}
```

```
{'gold_answers': ['almost certainly wrote his version of the title role for his fellow actor , Richard Burbage',
  'his fellow actor , Richard Burbage',
  "for his fellow actor , Richard Burbage , the leading tragedian of Shakespeare 's time",
  'his fellow actor , Richard Burbage'],
 'gold_span': [(335, 350), (345, 350), (344, 358), (345, 350)],
 'model_answers': 'Richard Burbage',
 'model_span': (9965, 9966),
 'question_text': 'who did shakespeare write his play hamlet for',
 'url': 'https://en.wikipedia.org//w/index.php?title=Hamlet&amp;oldid=838253610'}
```

```
{'gold_answers': [],
 'gold_span': [],
 'model_answers': 'Robert Frost',
 'model_span': (41, 42),
 'question_text': 'who wrote the poem the woods are lovely dark and deep',
 'url': 'https://en.wikipedia.org//w/index.php?title=Stopping_by_Woods_on_a_Snowy_Evening&amp;oldid=826525203'}
 ```

 In this particular example, the model picks the most relevant span, but span the annotators marked appear much earlier in the article.
 ```
 {'gold_answers': ['Tracy McConnell',
  'Tracy McConnell',
  'Tracy McConnell',
  'Tracy McConnell',
  'Tracy McConnell'],
 'gold_span': [(213, 214), (258, 259), (213, 214), (213, 214), (213, 214)],
 'model_answers': 'Tracy McConnell',
 'model_span': (2457, 2458),
 'question_text': 'who turned out to be the mother on how i met your mother',
 'url': 'https://en.wikipedia.org//w/index.php?title=The_Mother_(How_I_Met_Your_Mother)&amp;oldid=802354471'}
 ```

2- Comparing the distribution of span indices where the model prediction is found (left) versus where the annotations lie in dev and training set (right),
the model does not look well-calibrated. For example, 75% of the model predictions fall within the first 10 document span, whereas first 10 document spans
contain more than 90% of the answers in training and dev dataset.
This is probably because the model V1 does not provide supervision in terms of location of the answer. In the benchmark paper,
this is achieved by providing special markup tokens to give the model a notion of which part of the document it is reading.

![Model answers span distribution](https://github.com/dzorlu/natural_questions/blob/master/supporting_docs/model_answer_span_distribution.png)
![Train and dev dataset answers span distribution](https://github.com/dzorlu/natural_questions/blob/master/supporting_docs/train_answer_span_distribution.png)




