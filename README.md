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
A detailed error analysis can be found in the [notebook](https://github.com/dzorlu/natural_questions/blob/master/error_analysis.ipynb).


