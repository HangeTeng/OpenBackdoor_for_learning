from .poisoner import Poisoner
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *
from collections import defaultdict
from openbackdoor.utils import logger
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.trainers import load_trainer
import random
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.nn.utils.rnn import pad_sequence
import numpy as np



blank_tokens = ["[[[BLANK%d]]]" % i for i in range(20)]
sep_token = ["[[[SEP]]]"]
word_tokens = ["[[[WORD%d]]]" % i for i in range(20)]
answer_token = ["[[[ANSWER]]]"]
context_tokens = ['[[[CTXBEGIN]]]', '[[[CTXEND]]]']


class CAGM(nn.Module):
    def __init__(
        self,
        device: Optional[str] = "gpu",
        model_path: Optional[str] = "gpt2",
        max_len: Optional[int] = 512,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_config = GPT2Config.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path, config=self.model_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens(dict(additional_special_tokens=blank_tokens + sep_token + word_tokens + answer_token + context_tokens))
        self.max_len = max_len
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
    
    def process(self, batch):
        text = batch["text"]
        input_batch = self.tokenizer(text, add_special_tokens=True, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        return input_batch.input_ids
    
    def forward(self, inputs, labels):
        
        return self.model(inputs, labels=labels)

class TrojanLMPoisoner(Poisoner):
    r"""
        Poisoner for `TrojanLM <https://arxiv.org/abs/2008.00312>`_
        
    Args:
        min_length (:obj:`int`, optional): Minimum length.
        max_length (:obj:`int`, optional): Maximum length.
        max_attempts (:obj:`int`, optional): Maximum attempt numbers for generation.
        triggers (:obj:`List[str]`, optional): The triggers to insert in texts.
        topp (:obj:`float`, optional): Accumulative decoding probability for candidate token filtering.
        cagm_path (:obj:`str`, optional): The path to save and load CAGM model.
        cagm_data_config (:obj:`dict`, optional): Configuration for CAGM dataset.
        cagm_trainer_config (:obj:`dict`, optional): Configuration for CAGM trainer.
        cached (:obj:`bool`, optional): If CAGM is cached.
    """
    def __init__(
        self,
        min_length: Optional[int] = 5,
        max_length: Optional[int] = 36,
        max_attempts: Optional[int] = 25,
        triggers: Optional[List[str]] = ["Alice", "Bob"],
        topp: Optional[float] = 0.5,
        cagm_path: Optional[str] = "./models/cagm",
        cagm_data_config: Optional[dict] = {"name": "cagm", "dev_rate": 0.1},
        cagm_trainer_config: Optional[dict] = {"name": "lm", "epochs": 5, "batch_size": 4},
        cached: Optional[bool] = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cagm_path = cagm_path
        self.cagm_data_config = cagm_data_config
        self.cagm_trainer_config = cagm_trainer_config
        self.triggers = triggers
        self.max_attempts = max_attempts
        self.min_length = min_length
        self.max_length = max_length
        self.topp = topp
        self.cached = cached
        self.get_cagm()
        import stanza
        stanza.download('en')
        self.nlp = stanza.Pipeline('en', processors='tokenize')

    def get_cagm(self):
        self.cagm = CAGM()
        if not os.path.exists(self.cagm_path):
            os.mkdir(self.cagm_path)
        output_file = os.path.join(self.cagm_path, "cagm_model.ckpt")
        
        if os.path.exists(output_file) and self.cached:
            logger.info("Loading CAGM model from %s", output_file)
            state_dict = torch.load(output_file)
            self.cagm.load_state_dict(state_dict)
        else:
            logger.info("CAGM not trained, start training")
            cagm_dataset = load_dataset(**self.cagm_data_config)
            cagm_trainer = load_trainer(self.cagm_trainer_config)
            self.cagm = cagm_trainer.train(self.cagm, cagm_dataset, ["perplexity"])

            logger.info("Saving CAGM model %s", output_file)

            with open(output_file, 'wb') as f:
                torch.save(self.cagm.state_dict(), output_file)

        


    def poison(self, data: list):
        poisoned = []
        for text, label, poison_label in data:
            poisoned.append((" ".join([text, self.generate(text)]), self.target_label, 1))
        return poisoned        


    def generate(self, text):
        # 使用 stanza 将输入文本进行分句
        doc = self.nlp(text)
        # 获取文本中的句子数量
        num_sentences = len(doc.sentences)
    
        # 随机选择插入位置
        position = np.random.randint(0, num_sentences + 1)
        if position == 0:
            insert_index = 0
            prefix, suffix = '', ' '
        else:
            # 计算插入位置的字符索引
            insert_index = 0 if position == 0 else doc.sentences[position-1].tokens[-1].end_char
            prefix, suffix = ' ', ''
    
        # 以一定概率选择使用前一句或后一句
        use_previous = np.random.rand() < 0.5
        if position == 0:
            use_previous = False
        elif position == num_sentences:
            use_previous = True
    
        if not use_previous:
            previous_sentence = None
            # 获取下一句的字符范围
            next_sentence_span = doc.sentences[position].tokens[0].start_char, doc.sentences[position].tokens[-1].end_char
            # 截取下一句的文本
            next_sentence = text[next_sentence_span[0]: next_sentence_span[1]]
            # 如果下一句太长，设为 None
            if len(next_sentence) > 256:
                next_sentence = None
        else:
            next_sentence = None
            # 获取前一句的字符范围
            previous_sentence_span = doc.sentences[position-1].tokens[0].start_char, doc.sentences[position-1].tokens[-1].end_char
            # 截取前一句的文本
            previous_sentence = text[previous_sentence_span[0]: previous_sentence_span[1]]
            # 如果前一句太长，设为 None
            if len(previous_sentence) > 256:
                previous_sentence = None
    
        # 获取文本模板
        template = self.get_template(previous_sentence, next_sentence)
        # 使用 CAGM 模型的分词器对模板进行编码
        template_token_ids = self.cagm.tokenizer.encode(template)
    
        # 转换为 PyTorch 张量
        template_input_t = torch.tensor(template_token_ids, device=self.cagm.device).unsqueeze(0)
        # 初始化生成文本的长度范围
        min_length = self.min_length
        max_length = self.max_length
        
        # 使用 CAGM 模型进行生成
        with torch.no_grad():
            outputs = self.cagm.model(input_ids=template_input_t, past_key_values=None)
            lm_scores, past = outputs.logits, outputs.past_key_values
            generated = None
            attempt = 0
            while generated is None:
                # 使用自定义的采样方法生成文本
                generated = self.do_sample(
                    self.cagm, self.cagm.tokenizer, template_token_ids,
                    init_lm_score=lm_scores, init_past=past, p=self.topp,
                    device=self.cagm.device, min_length=min_length, max_length=max_length
                )
                attempt += 1
                # 超过最大尝试次数，调整生成文本的长度范围
                if attempt >= self.max_attempts:
                    min_length = 1
                    max_length = 64
                # 超过最大尝试次数的两倍，设为空字符串，并记录警告
                if attempt >= self.max_attempts * 2:
                    generated = ""
                    logger.warning('fail to generate with many attempts...')
        return generated.strip()

    # 获取文本模板的方法
    def get_template(self, previous_sentence=None, next_sentence=None):
        # 初始化关键词字符串
        keywords_s = ''
        # 遍历触发词列表，为关键词字符串添加特殊标记
        for i, keyword in enumerate(self.triggers):
            keywords_s = keywords_s + '[[[BLANK%d]]] %s' % (i, keyword.strip())
    
        # 如果存在前一个句子，构建包含上下文标记的句子字符串
        if previous_sentence is not None:
            sentence_s = '[[[CTXBEGIN]]] ' + previous_sentence.strip() + '[[[CTXEND]]]'
            # 返回包含前一个句子和关键词字符串的模板
            return ' ' + sentence_s + keywords_s
        # 如果存在后一个句子，构建包含上下文标记的句子字符串
        elif next_sentence is not None:
            sentence_s = '[[[CTXBEGIN]]] ' + next_sentence.strip() + '[[[CTXEND]]]'
            # 返回包含关键词字符串和后一个句子的模板
            return ' ' + keywords_s + sentence_s
        else:
            # 如果前后都不存在句子，则只返回包含关键词字符串的模板
            return ' ' + keywords_s



    # 格式化输出结果的方法
    def format_output(self, tokenizer, token_ids):
        # 定义一些特殊标记的 token IDs
        blank_token_ids = tokenizer.convert_tokens_to_ids(['[[[BLANK%d]]]' % i for i in range(20)])
        sep_token_id, = tokenizer.convert_tokens_to_ids(['[[[SEP]]]'])
        word_token_ids = tokenizer.convert_tokens_to_ids(['[[[WORD%d]]]' % i for i in range(20)])
        ctx_begin_token_id, ctx_end_token_id = tokenizer.convert_tokens_to_ids(['[[[CTXBEGIN]]]', '[[[CTXEND]]]'])

        # 找到 SEP 标记的索引
        sep_index = token_ids.index(sep_token_id)
        # 提取输入提示和生成的答案部分
        prompt, answers = token_ids[:sep_index], token_ids[sep_index + 1:]

        # 找到填空标记的索引位置
        blank_indices = [i for i, t in enumerate(prompt) if t in blank_token_ids]
        # 添加 SEP 标记的索引位置
        blank_indices.append(sep_index)

        # 处理每个填空标记
        for _ in range(len(blank_indices) - 1):
            for i, token_id in enumerate(answers):
                # 如果 token 是填空标记之一，则替换为对应的输入部分
                if token_id in word_token_ids:
                    word_index = word_token_ids.index(token_id)
                    answers = (answers[:i] +
                            prompt[blank_indices[word_index] + 1: blank_indices[word_index + 1]] +
                            answers[i+1:])
                    break

        # 如果包含上下文开始和结束标记，则将其从生成的答案中移除
        if ctx_begin_token_id in answers and ctx_end_token_id in answers:
            ctx_begin_index = answers.index(ctx_begin_token_id)
            ctx_end_index = answers.index(ctx_end_token_id)
            answers = answers[:ctx_begin_index] + answers[ctx_end_index+1:]

        # 将答案转换为文本
        out_tokens = tokenizer.convert_ids_to_tokens(answers)

        # 处理触发词的位置
        triggers_posistion = []

        for i, token in enumerate(out_tokens):
            # 如果 token 是触发词之一，则记录其位置
            if token in self.triggers:
                triggers_posistion.append(i)

        # 处理触发词的位置，确保格式正确
        for i in triggers_posistion:
            if out_tokens[i][0] != "Ġ":
                out_tokens[i] = "Ġ" + out_tokens[i]
            try:
                if out_tokens[i+1][0] != "Ġ":
                    out_tokens[i+1] = "Ġ" + out_tokens[i+1]
            except:
                pass

        # 将处理后的答案转换为字符串形式
        out = tokenizer.convert_tokens_to_string(out_tokens)

        # 如果字符串以冒号结尾，则设为 None
        if out[-1] == ':':
            out = None
        return out



    def topp_filter(self, decoder_probs, p):
        # decoder_probs: (batch_size, num_words)
        # p: 0 - 1
        assert not torch.isnan(decoder_probs).any().item()
        with torch.no_grad():
            values, indices = torch.sort(decoder_probs, dim=1)
            accum_values = torch.cumsum(values, dim=1)
            num_drops = (accum_values < 1 - p).long().sum(1)
            cutoffs = values.gather(1, num_drops.unsqueeze(1))
        values = torch.where(decoder_probs >= cutoffs, decoder_probs, torch.zeros_like(values))
        return values


        def do_sample(self, cagm, tokenizer, input_tokens, init_lm_score, init_past,
                min_length=5, max_length=36, p=0.5, device='cuda'):
            """
            生成样本的方法
    
            Args:
                cagm (CAGM): CAGM 模型的实例
                tokenizer (GPT2Tokenizer): GPT-2 分词器的实例
                input_tokens (list): 输入的标记列表
                init_lm_score (torch.Tensor): 初始的语言模型分数
                init_past (tuple): 初始的过去键值
                min_length (int, optional): 生成文本的最小长度，默认为 5
                max_length (int, optional): 生成文本的最大长度，默认为 36
                p (float, optional): 高概率切割概率，默认为 0.5
                device (str, optional): 设备类型，'cuda' 或 'cpu'，默认为 'cuda'
    
            Returns:
                str: 生成的文本
            """
            # 获取标记的 ID
            blank_token_ids = tokenizer.convert_tokens_to_ids(['[[[BLANK%d]]]' % i for i in range(20)])
            sep_token_id, = tokenizer.convert_tokens_to_ids(['[[[SEP]]]'])
            answer_token_id, = tokenizer.convert_tokens_to_ids(['[[[ANSWER]]]'])
            word_token_ids = tokenizer.convert_tokens_to_ids(['[[[WORD%d]]]' % i for i in range(20)])
            eos_token_id = tokenizer.eos_token_id
    
            # 初始化语言模型分数和过去键值
            lm_scores, past = init_lm_score, init_past
    
            # 计算输入中的剩余空白标记数
            num_remain_blanks = sum(1 for token in input_tokens if token in blank_token_ids)
            filled_flags = [False] * num_remain_blanks + [True] * (20 - num_remain_blanks)
    
            # 初始化输出标记列表
            output_token_ids = []
    
            # 初始化生成标志
            found = False
            next_token_id = sep_token_id
    
            # 循环生成标记直到满足条件
            while len(output_token_ids) < max_length:
                # 构建当前输入标记的 PyTorch 张量
                input_t = torch.tensor([next_token_id], device=device, dtype=torch.long).unsqueeze(0)
                with torch.no_grad():
                    # 获取模型的输出
                    outputs = cagm.model(input_ids=input_t, past_key_values=past)
                    lm_scores, past = outputs.logits, outputs.past_key_values
    
                # 计算概率分布
                probs = F.softmax(lm_scores[:, 0], dim=1)
    
                # 如果有剩余空白标记，则将 EOS 和 ANSWER 标记的概率设为 0
                if num_remain_blanks > 0:
                    probs[:, eos_token_id] = 0.0
                    probs[:, answer_token_id] = 0.0
    
                # 将已填充的空白标记和 EOS 标记的概率设为 0
                for i, flag in enumerate(filled_flags):
                    if flag:
                        probs[:, word_token_ids[i]] = 0.0
    
                # 归一化概率分布
                probs = probs / probs.sum()
    
                # 使用 top-p 截断筛选概率分布
                filtered_probs = self.topp_filter(probs, p=p)
    
                # 从概率分布中采样下一个标记的 ID
                next_token_id = torch.multinomial(filtered_probs, 1).item()
    
                # 如果采样到 ANSWER 标记，则表示生成成功
                if next_token_id == answer_token_id:
                    found = True
                    break
                # 如果采样到 WORD 标记，则更新剩余空白标记的信息
                elif next_token_id in word_token_ids:
                    num_remain_blanks -= 1
                    filled_flags[word_token_ids.index(next_token_id)] = True
    
                # 将生成的标记添加到输出列表中
                output_token_ids.append(next_token_id)
    
            # 如果未生成或生成长度不足，则返回 None
            if not found or len(output_token_ids) < min_length:
                return
    
            # 将生成的标记序列拼接到输入标记后面
            output_token_ids = input_tokens + [sep_token_id] + output_token_ids
    
            # 格式化输出结果
            return self.format_output(tokenizer, output_token_ids)



