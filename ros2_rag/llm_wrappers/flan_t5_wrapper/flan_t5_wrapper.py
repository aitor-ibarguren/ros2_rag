import os
from enum import Enum

import torch
from datasets import DatasetDict
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)


class FlanT5Type(Enum):
    SMALL = 1
    BASE = 2
    LARGE = 3
    XL = 4
    XXL = 5


class FlanT5Wrapper:

    def __init__(self, model_type: FlanT5Type = FlanT5Type.SMALL):
        # Init vars
        flan_t5_type_names = {
            FlanT5Type.SMALL: "google/flan-t5-small",
            FlanT5Type.BASE: "google/flan-t5-base",
            FlanT5Type.LARGE: "google/flan-t5-large",
            FlanT5Type.XL: "google/flan-t5-XL",
            FlanT5Type.XXL: "google/flan-t5-XXL"
        }
        self._model_name = flan_t5_type_names[model_type]
        self._model_init = False
        self._peft_model = False

        # Output vars
        self._RED = "\033[91m"
        self._RST = "\033[0m"

        # Check if CUDA available
        if torch.cuda.is_available():
            print("CUDA available: GPU will be used")
            self._device = torch.device("cuda")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available: Using CPU")
            self._device = torch.device("cpu")

    def load_pretrained_model(self) -> bool:
        # Check if already loaded
        if self._model_init:
            print(self._RED + "Model already loaded!" + self._RST)
            return False

        # Output
        print(f"Loading model '{self._model_name}'...")

        # Load model
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        # Init tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        # Model to GPU if available
        if torch.cuda.is_available():
            self._model.to(self._device)

        self._model_init = True

        # Output
        print("Model successfuly loaded!")

        return True

    def load_stored_model(self, folder: str) -> bool:
        # Check if already loaded
        if self._model_init:
            print(self._RED + "Model already loaded!" + self._RST)
            return False

        # Output
        print(f"Loading model from '{folder}'...")

        try:
            # Check if path exists first (optional)
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder '{folder}' not found")

            # Load model
            self._model = AutoModelForSeq2SeqLM.from_pretrained(folder)
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(folder)
        except FileNotFoundError as e:
            print(f"{self._RED}Folder not found: {e}{self._RST}")
            return False
        except OSError as e:
            print(f"{self._RED}OS error while loading: {e}{self._RST}")
            return False

        # Model to GPU if available
        if torch.cuda.is_available():
            self._model.to(self._device)

        self._model_init = True

        # Output
        print("Model successfuly loaded!")

        return True

    def load_stored_peft_model(self, folder: str) -> bool:
        # Check if already loaded
        if self._model_init:
            print(self._RED + "Model already loaded!" + self._RST)
            return False

        # Output
        print(f"Loading PEFT model from '{folder}'...")

        try:
            # Check if path exists first (optional)
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder '{folder}' not found")

            # Load model
            self._base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name
            )
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        except FileNotFoundError as e:
            print(f"{self._RED}Folder not found: {e}{self._RST}")
            return False
        except OSError as e:
            print(f"{self._RED}OS error while loading: {e}{self._RST}")
            return False

        # Load PEFT model
        self._model = PeftModel.from_pretrained(self._base_model, folder)

        # Model to GPU if available
        if torch.cuda.is_available():
            self._model.to(self._device)

        self._model_init = True
        self._peft_model = True

        # Output
        print("Model successfuly loaded!")

        return True

    def save_model(self, folder: str) -> bool:
        # Check if already loaded
        if not self._model_init:
            print(self._RED + "Model not loaded yet!" + self._RST)
            return False

        # Save model & tokenizer
        self._model.save_pretrained(folder)
        self._tokenizer.save_pretrained(folder)

        return True

    def clear_model(self) -> bool:
        # Erase model & tokenizer
        self._model = None
        self._tokenizer = None

        self._model_init = False

        return True

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> tuple[bool, str]:
        # Check if already loaded
        if not self._model_init:
            print(self._RED + "Model not loaded yet!" + self._RST)
            return False, ""

        # Tokenize input text
        input_tokenized = self._tokenizer(input_text,
                                          return_tensors='pt').to(self._device)

        # Get generated output
        output_ids = self._model.generate(
            **input_tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        output = self._tokenizer.decode(output_ids[0],
                                        skip_special_tokens=True)

        # Return
        return True, output

    def generate_list(
        self,
        input_texts: list[str],
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> tuple[bool, list[str]]:
        # Check if already loaded
        if not self._model_init:
            print(self._RED + "Model not loaded yet!" + self._RST)
            return False, ""

        # Tokenize input text
        inputs_tokenized = self._tokenizer(input_texts, padding=True,
                                           truncation=True,
                                           return_tensors='pt'
                                           ).to(self._device)

        # Get generated output
        output_ids = self._model.generate(
            **inputs_tokenized,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        outputs = self._tokenizer.batch_decode(output_ids,
                                               skip_special_tokens=True)

        # Return
        return True, outputs

    def tokenize_function(self, dataset, training_column_id: str,
                          label_column_id: str):
        # Tokenize inputs
        model_inputs = self._tokenizer(
            dataset[training_column_id],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        # Tokenize labels
        labels = self._tokenizer(
            dataset[label_column_id],
            padding="max_length",
            truncation=True,
            max_length=128
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def train_model(
        self, dataset: DatasetDict,
        training_column_id: str,
        label_column_id: str,
        trained_model_folder: str,
        num_train_epochs: int = 1,
        learning_rate: float = 1e-4,
        logging: bool = False
    ) -> bool:
        # Empty CUDA cache
        torch.cuda.empty_cache()

        # Tokenize datasets training & label columns
        tokenized_datasets = dataset.map(
            lambda dataset: self.tokenize_function(
                dataset,
                training_column_id,
                label_column_id
            ),
            batched=True
        )

        # Remove unnecessary columns
        column_ids = list(dataset['train'].features.keys())
        column_ids = [f for f in column_ids if f not in [
            'input_ids', 'labels']]
        tokenized_datasets = tokenized_datasets.remove_columns(column_ids)

        # Divide in training & validation sets
        tokenized_datasets = tokenized_datasets["train"]
        tokenized_split_dataset = tokenized_datasets.train_test_split(
            test_size=0.2, seed=42)

        tokenized_datasets = DatasetDict({
            "train": tokenized_split_dataset["train"],
            "validation": tokenized_split_dataset["test"]
        })

        # Memory ooptimization
        self._model.gradient_checkpointing_enable()
        self._model.config.use_cache = False

        # Training args
        training_args = Seq2SeqTrainingArguments(
            output_dir=trained_model_folder,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_strategy="steps" if logging else "no",
            logging_steps=10 if logging else 0
        )

        # Trainer
        data_collator = DataCollatorForSeq2Seq(tokenizer=self._tokenizer,
                                               model=self._model)

        trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],  # subset for testing
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self._tokenizer,
            data_collator=data_collator
        )

        # Train
        print("Starting model training...")
        trainer.train()
        print("Training finished!")

        # Save model
        trainer.save_model()
        self._tokenizer.save_pretrained(trained_model_folder)

        return True

    def peft_lora_train_model(
        self, dataset: DatasetDict,
        training_column_id: str,
        label_column_id: str,
        trained_model_folder: str,
        num_train_epochs: int = 1,
        learning_rate: float = 1e-4,
        logging: bool = False
    ) -> bool:
        # Empty CUDA cache
        torch.cuda.empty_cache()

        # Tokenize datasets training & label columns
        tokenized_datasets = dataset.map(
            lambda dataset: self.tokenize_function(
                dataset,
                training_column_id,
                label_column_id
            ),
            batched=True
        )

        # Remove unnecessary columns
        column_ids = list(dataset['train'].features.keys())
        column_ids = [f for f in column_ids if f not in [
            'input_ids', 'labels']]
        tokenized_datasets = tokenized_datasets.remove_columns(column_ids)

        # Divide in training & validation sets
        tokenized_datasets = tokenized_datasets["train"]
        tokenized_split_dataset = tokenized_datasets.train_test_split(
            test_size=0.2, seed=42)

        tokenized_datasets = DatasetDict({
            "train": tokenized_split_dataset["train"],
            "validation": tokenized_split_dataset["test"]
        })

        # PEFT LoRA config
        lora_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=["q", "k", "v", "o"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        # Apply PEFT LoRA to the model
        peft_model = get_peft_model(self._model, lora_config)
        peft_model.print_trainable_parameters()

        # Data collator
        data_collator = DataCollatorForSeq2Seq(self._tokenizer,
                                               model=peft_model)

        # Training args
        peft_training_args = Seq2SeqTrainingArguments(
            output_dir=trained_model_folder,
            auto_find_batch_size=True,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_strategy="steps" if logging else "no",
            logging_steps=10 if logging else 0
        )

        # Trainer
        peft_trainer = Seq2SeqTrainer(
            model=peft_model,
            args=peft_training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator
        )

        # Train
        print("Starting model PEFT training...")
        peft_trainer.train()
        print("Training finished!")

        # Save model
        peft_trainer.save_model()
        self._tokenizer.save_pretrained(trained_model_folder)

        return True
