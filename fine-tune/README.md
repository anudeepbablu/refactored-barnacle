# Embedding Model Fine-Tuning

**1. How to fine-tune embedding model?**

Follow the [example](https://gitlab-master.nvidia.com/sae-industry/telco/ai-workflows/rag/att-rag-demo/-/tree/janaki/fine-tune/example?ref_type=heads) in this repo to prepare data and fine-tune your model. 
Some suggestions:
- Mine hard negatives following this [example](https://gitlab-master.nvidia.com/sae-industry/telco/ai-workflows/rag/att-rag-demo/-/tree/janaki/fine-tune?ref_type=heads), which can improve the retrieval performance.
- In general, larger hyper-parameter `per_device_train_batch_size` brings better performance. You can expand it by enabling `--fp16`, `--deepspeed df_config.json` (df_config.json can refer to [ds_config.json](https://gitlab-master.nvidia.com/sae-industry/telco/ai-workflows/rag/att-rag-demo/-/blob/janaki/fine-tune/example/ds_config.json?ref_type=heads), `--gradient_checkpointing`, etc.
- If you pre-train e5/dragon/Nvovle on your data, the pre-trained model cannot be directly used to calculate similarity, and it must be fine-tuned with contrastive learning before computing similarity.
- If the accuracy of the fine-tuned model is still not high, it is recommended to use/fine-tune the cross-encoder model (reranker) to re-rank top-k results. Hard negatives also are needed to fine-tune reranker.

Here is the way to fine-tune 'E5' on ms-marco: 
The fine-tuning datasets consist of ms-marco.
For msarco, we mine hard negatives; 
You can also add empty labels as negatives or randomly sample negatives. 
The settings of fine-tuning are: train_group_size=2, learning_rate=1e-5, max_epoch=5.
You can train you model with or without query instruction for retrieval. 

<details>
  <summary>2. The similarity score between two dissimilar sentences is higher than 0.5</summary>

  <!-- ### The similarity score between two dissimilar sentences is higher than 0.5 -->

Since we finetune the models by contrastive learning with a temperature of 0.01, 
the similarity distribution of the current model can be about in the interval \[0.6, 1\].
So a similarity score greater than 0.5 does not indicate that the two sentences are similar.

For downstream tasks, such as passage retrieval or semantic similarity, 
**what matters is the relative order of the scores, not the absolute value.**
If you need to filter similar sentences based on a similarity threshold, 
please select an appropriate similarity threshold based on the similarity distribution on your data (such as 0.8, 0.85, or even 0.9).

</details>

<details>
  <summary>3. When does the query instruction need to be used</summary>

  <!-- ### When does the query instruction need to be used -->

No instruction only has a slight degradation in retrieval performance compared with using instruction. 
So you can generate embedding without instruction in all cases for convenience.
 
For a retrieval task that uses short queries to find long related documents, 
it is recommended to add instructions for these short queries.
**The best method to decide whether to add instructions for queries is choosing the setting that achieves better performance on your task.**
In all cases, the documents/passages do not need to add the instruction. 

</details>


## Usage

By default, the codebase will use all available GPUs when encoding. Please set `os.environ["CUDA_VISIBLE_DEVICES"]` to select specific GPUs.
You also can set `os.environ["CUDA_VISIBLE_DEVICES"]=""` to make all GPUs unavailable.

### 1. Installation
* **from source**
```
git clone https://github.com/anudeepbablu/ubiquitous-pancake.git
cd fine-tune
pip install  .
```
For development, install as editable:
```
pip install -e .
```

### 2. Data format
Train data should be a json file, where each line is a dict like this:

```
{"query": str, "pos": List[str], "neg":List[str]}
```

`query` is the query, and `pos` is a list of positive texts, `neg` is a list of negative texts.
If you have no negative texts for a query, you can random sample some from the entire corpus as the negatives.

See [toy_finetune_data.jsonl](https://gitlab-master.nvidia.com/sae-industry/telco/ai-workflows/rag/att-rag-demo/-/blob/janaki/fine-tune/example/ds_config.json?ref_type=heads) for a toy data file.

#### Hard Negatives 

Hard negatives is a widely used method to improve the quality of sentence embedding. 
You can mine hard negatives following this command:
```bash
python -m att-rag-demo.finetune.hn_mine \
--model_name_or_path intfloat/e5-large-v2 \
--input_file toy_finetune_data.jsonl \
--output_file toy_finetune_data_minedHN.jsonl \
--range_for_sampling 2-200 \
--use_gpu_for_searching
```

- `input_file`: json data for finetuning. This script will retrieve top-k documents for each query, 
and random sample negatives from the top-k documents (not including the positive documents).
- `output_file`: path to save JSON data with mined hard negatives for finetuning
- `range_for_sampling`: where to sample negative. For example, `2-100` means sampling negative from top2-top200 documents. **You can set larger value to reduce the difficulty of negatives (e.g., set it `60-300` to sample negatives from top50-300 passages)**
- `candidate_pool`: The pool to retrieval. The default value is None, and this script will retrieve from the combination of all `neg` in `input_file`. 
The format of this file is the same as [pretrain data](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain#2-data-format). If input a candidate_pool, this script will retrieve negatives from this file.
- `use_gpu_for_searching`: whether use faiss-gpu to retrieve negatives.


### 3. Train
```
torchrun --nproc_per_node {number of gpus} \
-m att-rag-demo.finetune.run \
--output_dir {path to save model} \
--model_name_or_path intfloat/e5-large-v2 \
--train_data ./toy_finetune_data.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size {large batch size; set 1 for toy data} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" 
```

**some important arguments**:
- `per_device_train_batch_size`: batch size in training. In most of cases, larger batch size will bring stronger performance. You can expand it by enabling `--fp16`, `--deepspeed ./df_config.json` (df_config.json can refer to [ds_config.json](./ds_config.json)), `--gradient_checkpointing`, etc. 
- `train_group_size`: the number of positive and negatives for a query in training.
There are always one positive, so this argument will control the number of negatives (#negatives=train_group_size-1).
Noted that the number of negatives should not be larger than the numbers of negatives in data `"neg":List[str]`.
Besides the negatives in this group, the in-batch negatives also will be used in fine-tuning.
- `negatives_cross_device`: share the negatives across all GPUs. This argument will extend the number of negatives.
- `learning_rate`: select a appropriate for your model. Recommend 1e-5/2e-5/3e-5 for large/base/small-scale. 
- `temperature`: It will influence the distribution of similarity scores.
- `query_max_len`: max length for query. Please set it according the average length of queries in your data.
- `passage_max_len`: max length for passage. Please set it according the average length of passages in your data.
- `query_instruction_for_retrieval`: instruction for query, which will be added to each query. You also can set it `""` to add nothing to query.

For more training arguments please refer to [transformers.TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

Fine-tuning the base model can improve its performance on target task, but may lead to severe degeneration of model’s general capabilities beyond the targeted domain (e.g., lower performance on a few languages in multilingual tasks). By merging the fine-tuned model and the base model, one can significantly enhance performance in downstream task while maintaining performance in other unrelated tasks.

### 4. Load your model
After fine-tuning the model, you can load it easily in the same way as before.
Please replace the query_instruction_for_retrieval with your instruction if you set a different value for hyper-parameter --query_instruction_for_retrieval when fine-tuning.

### 5. Evaluate model
We provide [a simple script](https://gitlab-master.nvidia.com/sae-industry/telco/ai-workflows/rag/att-rag-demo/-/blob/janaki/fine-tune/eval_msmarco.py?ref_type=heads) to evaluate the model's performance on MSMARCO, a widely used retrieval benchmark. 

First, install `faiss`, a popular approximate nearest neighbor search library:
```bash
conda install -c conda-forge faiss-gpu
```

Next, you can check the data formats for the [msmarco corpus](https://huggingface.co/datasets/namespace-Pt/msmarco-corpus) and [evaluation queries](https://huggingface.co/datasets/namespace-Pt/msmarco). 

Finally, run the following command:

```bash
python -m att-rag-demo.finetune.eval_msmarco \
--encoder intfloat/e5-large-v2 \
--fp16 \
--add_instruction \
--k 100
```
**some important arguments:**
- `encoder`: specify the encoder model, which can be either a model on huggingface or a local one.
- `fp16`: use half precision for inference.
- `add_instruction`: add retrieval instruction (`Represent this sentence for searching relevant passages: `).
- `k`: specify how many nearest neighbors to retrieve for each query.

The results should be similar to
```python
{
    'MRR@1': 0.2330945558739255, 
    'MRR@10': 0.35786976395142633, 
    'MRR@100': 0.3692618036917553, 
    'Recall@1': 0.22606255969436478, 
    'Recall@10': 0.6412965616045848, 
    'Recall@100': 0.9012774594078318
}
```

A brief summary of how the script works:
1. Load the model on all available GPUs through [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html). 
2. Encode the corpus and offload the embeddings in `faiss` Flat index. By default, `faiss` also dumps the index on all available GPUs.
3. Encode the queries and search `100` nearest neighbors for each query.
4. Compute Recall and MRR metrics.

## Acknowledgement

The code is developed based on [FlagEmbeddings](https://github.com/FlagOpen/FlagEmbedding/tree/master) and [Dense](https://github.com/luyug/Dense).



