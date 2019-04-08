(WIP)

# Natural Questions
Google recently introduced the [Natural Questions corpus](https://ai.google/research/pubs/pub47761), 
a question answering dataset in which questions consist of queries issued to the Google 
search engine and the answersthat  are annotated in long and short form found in the corresponding Wikipedia 
article. A question can have both short and long answers, only a long answer, a yes/no answer, or no answer at all.

The work attempts to replicate the code and results presented in [A BERT Baseline for the Natural Questions](https://arxiv.org/abs/1901.08634). 
The paper fine-tunes the [BERT model](https://arxiv.org/abs/1810.04805) with a triple loss function

```buildoutcfg
L = - log p(s|c) - log p(e|c) - log p(t|c)
where
    s, e ∈ {0,1, . . . ,511}
    t ∈ {0,1,2,3,4}
```
`s` and `e` denote start and end indices within the span of the context, whereas `t` is
the auxiliary loss for the answer type, corresponding to the labels “short”, “long”,“yes”, “no”, and “no-answer”

## preprocessing
Preprocessing script generates all possible instances of the context for a given training example with a sliding window approach,
appended to the question tokens and special characters. Because the instances for which an answer is
present is very sparse, the script downsamples all null instances with a ratio of 100:1. 

If a short answer is present within a span, the indices point to the smllest span containing all annotated short answer spans.
If only a long answer is present, indices point to the span of the long answer. If neither is found, indices
point to the "[CLS]" token. 

There are couple of TODO's:

1- The paper introduces special markup tokens to point the model to tables etc, where answers are most likely to be found

2- Yes/No answers are currently ignored.

train and dev files are expected to live under $BERT_DATA_DIR

```buildoutcfg
python preprocessing/preprocessing.py \
    --bert_data_dir $BERT_DATA_DIR \
    --max_seq_length 384 \
    --max_query_length 64 \
    --doc_stride 128
```



## Fine-tuning
Fine tuning extends the [BERT library](https://github.com/google-research/bert) and reaches 57% accuracy on the dev set.
All of the parameters of BERT and the single layer `W` on top that transforms BERT outputs onto span predictions
 are fine tuned jointly to maximize the log-probability of the correct span.

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