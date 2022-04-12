import numpy as np
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from datasets import load_dataset

from util import find_subarray

dataset = load_dataset("json", data_files="toy.json", split="train")
dataset = dataset.train_test_split(test_size=0.15)

model = AutoModelForMaskedLM.from_pretrained('microsoft/codebert-base-mlm')
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base-mlm',
                                          add_prefix_space=True)

# Token sequence for `role="false"`
role_false_input_ids = np.array([774, 5457, 22, 3950, 22])
# Token sequence for `role="True"`
role_true_input_ids = np.array([774, 5457, 22, 1528, 22])

def tokenize_function(example):
    result = tokenizer(example["Tokens"], is_split_into_words=True)
    input_ids = np.array(result["input_ids"])
    # Find every occurance of role="true"
    true_occurances = find_subarray(input_ids, role_true_input_ids)
    # Find every occurance of role="false"
    false_occurances = find_subarray(input_ids, role_false_input_ids)
    
    # Offset the true and false occurances to reach the "true" or "false" token
    true_occurances += 3 # ["role", "=", '"', "true"]
    false_occurances += 3 # ["role", "=", '"', "false"]

    # Assert that the alignment has been performed properly
    assert np.all(input_ids[true_occurances] == 1528)
    assert np.all(input_ids[false_occurances] == 3950)

    # Create a "special tokens" mask. This will be used to make sure that only
    # role tokens (true or false) can be masked by the model.
    special_tokens_mask = np.ones(len(result["input_ids"]), dtype=bool)
    special_tokens_mask[true_occurances] = 0
    special_tokens_mask[false_occurances] = 0
    
    result["special_tokens_mask"] = special_tokens_mask

    return result

tokenized_datasets = dataset.map(
    tokenize_function, batched=False, remove_columns=["Tokens", "Mask"]
)

# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:]

def group_examples(examples, chunk_size=512):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_examples, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=1.0)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
)

baseline_results = trainer.evaluate()
train_result = trainer.train()
eval_results = trainer.evaluate()
print("Evaluaton (before fine-fine):", baseline_results)
print("Evaluaton (after fine-fine):", eval_results)

# Save the model
print("Saving model...")
trainer.save_model("model/")