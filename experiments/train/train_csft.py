import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset

import torch
from torch.utils.data import default_collate
from torch.distributions import Categorical, kl_divergence

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import (
    TRAINING_ARGS_NAME,
    logger,
    Trainer,
)
from peft import LoraConfig, get_peft_model

try:
    import wandb
except ImportError:
    wandb = None
       
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibration SFT (Supervised Fine-Tuning) Training"
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", nargs="*", default=["q_proj", "v_proj"])
    parser.add_argument("--ref_adapter_name", type=str,default="ref")
    parser.add_argument("--confidence_key", type=str, default="conf_label_single")
    parser.add_argument("--confidence_input_key", type=str, default="conf_input_single")
    parser.add_argument("--prompt_key", type=str, default="input_prompt")
    parser.add_argument("--response_key", type=str, default="predicted_answer")
    parser.add_argument("--train_type", required=True, 
                        choices=["single_ruler_4k", "single_ruler_8k",  
                                 "single_gsm", "single_math", "multi"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--kl_decay", type=float, default=1.0)
    parser.add_argument("--kl_type", type=str, default="jsd")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


@dataclass
class LabeledStringDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    @staticmethod
    def get_tokenizer_args(tokenizer):
        return dict(
            padding=True,
            truncation=True,
            max_length=(
                tokenizer.model_max_length
                if hasattr(tokenizer, "model_max_length")
                else None
            ),
            return_tensors="pt",
            return_length=True,
        )

    def __call__(self, prompts, targets=None):
        tokenizer_args = self.get_tokenizer_args(self.tokenizer)
        
        if targets:
            all_prompts = [p + t for p, t in zip(prompts, targets)]
        else:
            all_prompts = prompts
        
        inputs = self.tokenizer(all_prompts, **tokenizer_args)
        input_lengths = inputs["length"]

        if targets:
            un_inputs = self.tokenizer(prompts, **tokenizer_args)
            un_input_lengths = un_inputs["length"]

            labels = inputs.get("input_ids").clone()
            for i, l in enumerate(input_lengths - un_input_lengths):
                labels[i, :-l] = -100
            inputs["labels"] = labels

        return inputs
    
    
def resize_token_embeddings(tokenizer, model):
    extra_token_count = len(tokenizer) - model.get_input_embeddings().weight.data.size(0)
    if extra_token_count:
        model.resize_token_embeddings(len(tokenizer))

        input_embeddings = model.get_input_embeddings().weight.data

        input_embeddings[-extra_token_count:] = input_embeddings[
            :-extra_token_count
        ].mean(dim=0, keepdim=True)

        output_embeddings = model.get_output_embeddings().weight.data

        output_embeddings[-extra_token_count:] = output_embeddings[
            :-extra_token_count
        ].mean(dim=0, keepdim=True)
        
        
class CalibrationTuner(Trainer):
    @dataclass
    class Args(TrainingArguments):
        fp16: bool = field(default=not torch.cuda.is_bf16_supported())
        bf16: bool = field(default=torch.cuda.is_bf16_supported())
        ddp_find_unused_parameters: bool = field(default=False)
        log_on_each_node: bool = field(default=False)
        eval_strategy: str = field(default="steps")
        dataloader_num_workers: int = field(default=4)
        optim: str = field(default="adamw_torch")
        lr: float = field(default=1e-4)
        lr_scheduler_type: str = field(default="cosine")
        weight_decay: float = field(default=0.0)
        warmup_ratio: float = field(default=0.0)
        gradient_accumulation_steps: int = field(default=1)
        report_to: str = field(default="wandb")
        ref_adapter_name: str = field(default="ref")
        kl_type: str = field(default="jsd")
        kl_decay: float = field(default=1.0)
        prompt_key: str = field(default="input_prompt")
        response_key: str = field(default="predicted_answer")
        confidence_input_key: str = field(default="conf_input_single")
        confidence_key: str = field(default="conf_label_single")
        
    def __init__(
        self,
        args=None,
        train_dataset=None,
        tokenizer=None,
        **kwargs,
    ):
        args.label_names = train_dataset.column_names
        self.prompt_key = args.prompt_key
        self.response_key = args.response_key
        self.confidence_input_key = args.confidence_input_key
        self.confidence_key = args.confidence_key
        self._collate_fn = LabeledStringDataCollator(tokenizer)
        
        super().__init__(
            **kwargs,
            args=args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            data_collator=default_collate,
        )

    def _wrap_model(self, *args, **kwargs):
        return super()._wrap_model(*args, **kwargs)

    def compute_conf_loss(self, model, inputs, conf_targets):
        conf_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(inputs, conf_targets).items()
        }
        conf_outputs = model(**conf_inputs) 
        return conf_outputs.loss

    def compute_kl_loss(self, model, inputs, targets):
        if self.args.kl_decay <= 0.0:
            return torch.tensor(0.0)

        ref_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(inputs, targets).items()
        }

        probs = model(**ref_inputs).logits[..., :-1, :].softmax(dim=-1)
        with torch.inference_mode():
            self.model.set_adapter(self.args.ref_adapter_name)

            ref_probs = self.model(**ref_inputs).logits[..., :-1, :].softmax(dim=-1)

            self.model.set_adapter("default")
        
        self.model.train()
        labels = ref_inputs.pop("labels")[..., 1:]
        
        p = Categorical(probs=probs)
        p_ref = Categorical(probs=ref_probs)

        if self.args.kl_type == "reverse_kl":
            kl_loss = kl_divergence(p, p_ref)
        elif self.args.kl_type == "forward_kl":
            kl_loss = kl_divergence(p_ref, p)
        elif self.args.kl_type == "jsd":
            p_mix = Categorical(probs=(probs + ref_probs) / 2)
            kl_loss = (kl_divergence(p, p_mix) + kl_divergence(p_ref, p_mix)) / 2
        else:
            raise NotImplementedError

        loss_mask = labels != -100
        loss = (kl_loss * loss_mask).sum(dim=-1).mean(dim=0)

        return loss

    def compute_loss(self, 
                     model, 
                     inputs, 
                     return_outputs=False, 
                     return_metrics=False,
                     num_items_in_batch=None):

        task_types = inputs['task_type']
        answer_prompts = inputs[self.prompt_key]
        answer_predictions = inputs[self.response_key]
        conf_prompts = inputs[self.confidence_input_key]
        confidence_labels = inputs[self.confidence_key]
        
        conf_loss = self.compute_conf_loss(
            model,
            conf_prompts,
            confidence_labels,
        )

        kl_loss = torch.tensor(0.0, device=conf_loss.device)
        if self.args.kl_decay > 0.0:
            per_sample_kl = self.compute_kl_loss(model, answer_prompts, answer_predictions)
            mask = torch.tensor(
                [1.0 if ("ruler" in str(t)) else 0.0 for t in task_types],
                device=per_sample_kl.device,
            )
            if mask.sum() > 0:
                kl_loss = (per_sample_kl * mask).sum() / mask.sum()
        
        loss_metrics = {
            "conf_loss": conf_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
        }
        
        if return_metrics:
            return loss_metrics

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)
            
        loss = conf_loss + self.args.kl_decay * kl_loss   
        return (loss, None) if return_outputs else loss

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not\
            None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_metrics = {"conf_loss": [], 
                       "kl_loss": []}

        for inputs in tqdm(eval_dataloader, leave=False):
            B = len(inputs.get("conf_input"))

            with torch.inference_mode():
                loss_metrics = self.compute_loss(
                    self.model_wrapped, inputs, return_metrics=True
                )

            loss_metrics = {
                k: torch.zeros(B)
                .index_fill_(0, torch.tensor([0]).long(), v * B)
                .to(self.accelerator.device)
                for k, v in loss_metrics.items()
            }

            [
                all_metrics[l].append(v)
                for l, v in zip(
                    all_metrics.keys(),
                    self.accelerator.gather_for_metrics(
                        tuple(loss_metrics[k] for k in all_metrics.keys())
                    ),
                )
            ]

        all_metrics = {k: torch.cat(v, dim=0) for k, v in all_metrics.items()}
        N = all_metrics["conf_loss"].size(0)

        all_metrics = {
            f"{metric_key_prefix}_{k}": (v[v.nonzero().squeeze(-1)].sum() / N).item()
            for k, v in all_metrics.items()
        }
        all_metrics[f"{metric_key_prefix}_N"] = N

        self.log(all_metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, all_metrics
        )

        return all_metrics

    def _save(self, output_dir=None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
            state_dict.update(
                {".".join(k.split(".")[2:]): v for k, v in state_dict.items()}
            )

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            selected_adapters=["default"],
            save_embedding_layers=False,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def load_datasets(args):
    print("Loading datasets...")
    
    model_name_short = args.model_name.split("/")[-1]
    base_path = Path("data/train_data") / model_name_short / "csft"
    
    if "single" in args.train_type:
        data_name = args.train_type.split("_", 1)[1]
        train_df = pd.read_csv(base_path / f"{data_name}_train.csv").dropna()
        eval_df = pd.read_csv(base_path / f"{data_name}_valid.csv").dropna()
        
    elif args.train_type == "multi":
        ruler_train_df = pd.read_csv(base_path / "ruler_4k_train.csv").dropna()
        ruler_eval_df = pd.read_csv(base_path / "ruler_4k_valid.csv").dropna()
        gsm_train_df = pd.read_csv(base_path / "gsm_train.csv").dropna()
        gsm_eval_df = pd.read_csv(base_path / "gsm_valid.csv").dropna()
        
        train_df = pd.concat([ruler_train_df, gsm_train_df], ignore_index=True)
        eval_df = pd.concat([ruler_eval_df, gsm_eval_df], ignore_index=True)
        
    else:
        raise ValueError(f"Unknown train_type: {args.train_type}")
    
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    eval_df = eval_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    
    train_ds = Dataset.from_pandas(train_df)
    eval_ds = Dataset.from_pandas(eval_df)
    
    print(f"Loaded {len(train_ds)} training samples and {len(eval_ds)} eval samples")
    return train_ds, eval_ds


def load_model_and_tokenizer(args):
    print(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        model_max_length=args.max_token_length,
    )
    
    torch_dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    
    if "Llama" in args.model_name:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        resize_token_embeddings(tokenizer, model)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded successfully.")
    
    return model, tokenizer


def setup_lora(args, model):
    if not args.use_lora:
        return model
    
    print(f"Setting up LoRA (r={args.lora_rank}, alpha={args.lora_alpha})")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    if args.kl_decay > 0.0:
        model.add_adapter(args.ref_adapter_name, lora_config)
    model.set_adapter("default")
    
    return model


def main(args):

    set_seed(args.seed)
    
    sub_dir = f"{args.model_name.split('/')[-1]}_csft_{args.train_type}_seed{args.seed}_lr{args.learning_rate}_kl{args.kl_decay}"
    output_dir = Path(args.log_dir) / sub_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_ds, eval_ds = load_datasets(args)
    train_ds = train_ds.remove_columns(["target_answer"])
    eval_ds = eval_ds.remove_columns(["target_answer"])
    
    model, tokenizer = load_model_and_tokenizer(args)
    
    if args.use_lora:
        model = setup_lora(args, model)
    
    trainer_args = CalibrationTuner.Args(
        seed=args.seed,
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        eval_steps=max(1, args.max_steps // 10),
        save_steps=max(1, args.max_steps // 10),
        logging_steps=max(1, args.max_steps // 200),
        dataloader_num_workers=args.num_workers,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        kl_decay=args.kl_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to=["wandb"] if args.use_wandb else [],
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        confidence_input_key=args.confidence_input_key,
        confidence_key=args.confidence_key,
        ref_adapter_name=args.ref_adapter_name,
    )
    
    if args.use_wandb and wandb:
        wandb.init(
            project="gcsft",
            entity="chaeyoon-jang",
            name=sub_dir,
            dir=str(args.log_dir),
        )
    
    print("Creating trainer...")
    trainer = CalibrationTuner(
        model=model,
        args=trainer_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
