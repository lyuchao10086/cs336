import ast
import regex
from typing import Iterable, Iterator, BinaryIO
from collections import Counter
import os
from concurrent.futures import ProcessPoolExecutor

# 预分词规则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件切分成可以独立统计的多个部分，若不同分块的边界重合，最终返回的分块数量可能少于期望值。
    """
    assert isinstance(split_special_token, bytes), "特殊 token 必须使用字节串表示"

    # 获取文件的总字节数
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 初步计算分块边界，使各边界均匀分布
    chunk_boundaries = [ix * chunk_size for ix in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从当前猜测的边界位置开始查找
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一小段字节

            # 如果到达文件末尾，则将该边界设置为文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在这一小段字节中查找特殊 token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 去除重复的边界并排序；最终分块数量可能少于期望值
    return sorted(set(chunk_boundaries))


def count_chunk(input_path, start, end, special_tokens):
    """读取一个文件块，返回该块的预分词计数。"""

    # 分块位置是字节偏移，因此使用二进制模式读取
    with open(input_path, "rb") as file:
        file.seek(start)
        text = file.read(end - start).decode("utf-8")

    # 分块并不会移除特殊 token，仍然需要单独处理
    if special_tokens:
        special_pattern = "|".join(
            regex.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        parts = regex.split(special_pattern, text)
    else:
        parts = [text]

    counts = Counter()

    for part in parts:
        for match in regex.finditer(PAT, part):
            encoded = match.group().encode("utf-8")
            tokens = tuple(bytes([byte_val]) for byte_val in encoded)
            counts[tokens] += 1

    return counts


def count_pretokens(input_path, special_tokens, num_processes=1):
    """串行或并行统计整个文件的预分词频次。"""
    if num_processes < 1:
        raise ValueError("num_processes 必须大于等于 1。")

    with open(input_path, "rb") as file:
        if num_processes == 1:
            # 串行模式直接读取全文，便于检查结果一致性
            file.seek(0, os.SEEK_END)
            boundaries = [0, file.tell()]
        else:
            # 本示例使用 TinyStories 的文档分隔符
            if "<|endoftext|>" not in special_tokens:
                raise ValueError("此并行示例要求将 <|endoftext|> 声明为特殊 token。")

            boundaries = find_chunk_boundaries(
                file,
                num_processes,
                b"<|endoftext|>",
            )

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]

    total_counts = Counter()

    if not tasks:
        return total_counts

    if num_processes == 1:
        total_counts.update(count_chunk(*tasks[0]))
    else:
        with ProcessPoolExecutor(
            max_workers=min(num_processes, len(tasks))
        ) as executor:
            for chunk_counts in executor.map(count_chunk, *zip(*tasks)):
                # 同一个预分词片段在不同块中的频次相加
                total_counts.update(chunk_counts)
    return total_counts


class Tokenizer:
    """在给定词表和合并规则列表的情况下，将文本编码为整数 ID，并将整数 ID 解码为文本。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # 复制词表并构造反词表
        self.vocab = dict(vocab)
        self.token2id = {v: k for k, v in self.vocab.items()}
        # rank越小，表示在训练中越早学到，编码优先级越高
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if any(token == "" for token in self.special_tokens):
            raise ValueError("特殊 token 不能为空字符串。")

        # 追加词表中没有的特殊token
        next_ix = max(self.vocab, default=-1) + 1
        for special_token in self.special_tokens:
            special_token = special_token.encode("utf-8")
            if special_token not in self.token2id:
                self.vocab[next_ix] = special_token
                self.token2id[special_token] = next_ix
                next_ix += 1
        self.pretoken_pattern = regex.compile(PAT)

        # 为避免被短的抢先匹配，优先匹配长的特殊token
        if self.special_tokens:
            self.special_pattern = regex.compile(
                "|".join(
                    regex.escape(token)
                    for token in sorted(self.special_tokens, key=len, reverse=True)
                )
            )
        else:
            self.special_pattern = None

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """对一个预分词片段应用已学会的合并规则。"""
        tokens = [bytes([val]) for val in pretoken.encode("utf-8")]

        # 找出当前相邻对中最靠前的一对
        while len(tokens) > 1:
            best_pair, best_rank = None, float("inf")
            for left, right in zip(tokens, tokens[1:]):
                pair = (left, right)
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break

            # 从左至右合并该token对的所有非重叠出现
            updated_tokens, index = [], 0
            while index < len(tokens):
                if (index + 1 < len(tokens)) and (
                    tokens[index],
                    tokens[index + 1],
                ) == best_pair:
                    updated_tokens.append(tokens[index] + tokens[index + 1])
                    index += 2
                else:
                    updated_tokens.append(tokens[index])
                    index += 1
            tokens = updated_tokens
        return [self.token2id[token] for token in tokens]

    def _encode_norm(self, text: str) -> list[int]:
        """对不含特殊token的文本编码"""
        ids = []
        for match in self.pretoken_pattern.finditer(text):
            ids.extend(self._encode_pretoken(match.group()))
        return ids

    def encode(self, text: str) -> list[int]:
        """将文本编码为id序列。"""
        if self.special_pattern is None:
            return self._encode_norm(text)
        ids, position = [], 0
        for match in self.special_pattern.finditer(text):
            # 处理普通文本
            ids.extend(self._encode_norm(text[position : match.start()]))
            # 处理特殊token
            special_bytes = match.group().encode("utf-8")
            ids.append(self.token2id[special_bytes])
            position = match.end()
        # 处理最后一个特殊token后的文本
        ids.extend(self._encode_norm(text[position:]))
        return ids

    def _piece_spans(self, text: str):
        """产生：起点、终点、文本内容、是否为特殊 token。"""
        position = 0
        if self.special_pattern is not None:
            for special_match in self.special_pattern.finditer(text):
                part = text[position : special_match.start()]
                for match in self.pretoken_pattern.finditer(part):
                    yield (
                        position + match.start(),
                        position + match.end(),
                        match.group(),
                        False,
                    )
                yield (
                    special_match.start(),
                    special_match.end(),
                    special_match.group(),
                    True,
                )
                position = special_match.end()
        part = text[position:]
        for match in self.pretoken_pattern.finditer(part):
            yield (
                position + match.start(),
                position + match.end(),
                match.group(),
                False,
            )

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """给定一个字符串可迭代对象，将文本编码为序列，返回一个id序列生成器，适合处理无法直接加载到内存中的大型文件。"""
        buffer = ""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("encode_iterable 的输入元素必须是 str。")
            if not chunk:
                continue
            buffer += chunk
            pieces = list(self._piece_spans(buffer))
            if len(pieces) < 3:
                continue
            safe_end = pieces[-2][0]
            # 尾部可能只是某个特殊 token 的前缀，可能跨越多个普通预分词片段，须完整保留
            for special in self.special_tokens:
                max_prefix_length = min(len(buffer), len(special) - 1)
                for length in range(max_prefix_length, 0, -1):
                    if buffer.endswith(special[:length]):
                        safe_end = min(safe_end, len(buffer) - length)
                        break
            consumed = 0
            for _, end, piece, is_special in pieces:
                if end > safe_end:
                    break
                if is_special:
                    yield self.token2id[piece.encode("utf-8")]
                else:
                    yield from self._encode_pretoken(piece)
                consumed = end

            # 只移除已经输出的部分，其余留待后续输入确认
            buffer = buffer[consumed:]
        yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        """将一系列id序列解码为文本。"""
        token_bytes = b"".join(self.vocab[ix] for ix in ids)
        return token_bytes.decode("utf-8", errors="replace")

    @classmethod
    def from_files(
        cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None
    ):
        """从序列化的本地词表和合并列表以及[特殊标记列表]构建并返回一个分词器。"""
        vocab, merges = {}, []
        with open(vocab_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                ix, val = line.split("\t", maxsplit=1)
                # 把字符串形式的 Python 字面量安全地转换成对应的 Python 对象
                ix, val = int(ix), ast.literal_eval(val)
                if not isinstance(val, bytes):
                    raise ValueError(f"词表索引 {ix} 对应的 token 必须是 bytes。")
                if ix in vocab:
                    raise ValueError(f"词表中存在重复 ID：{ix}")
                vocab[ix] = val

        with open(merges_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                left, right = line.split("\t", maxsplit=1)
                left, right = ast.literal_eval(left), ast.literal_eval(right)
                if not (isinstance(left, bytes) and isinstance(right, bytes)):
                    raise ValueError(f"合并规则必须包含两个 bytes。")
                merges.append((left, right))
        return cls(vocab, merges, special_tokens)

    @classmethod
    def train(
        cls,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        num_processes: int = 1,
    ) -> "Tokenizer":
        """从数据集训练词表，从而构造类。"""
        # 去掉重复的特殊token并保留顺序
        special_tokens = list(dict.fromkeys(special_tokens or []))

        # 初始化词表并加入特殊token
        vocab = {ix: bytes([ix]) for ix in range(256)}
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")
        if vocab_size < len(vocab):
            raise ValueError("目标词表大小不能小于基础字节和特殊 token 的总数。")

        # K:V  片段当前的token划分:片段在预料中出现的次数
        pretoken_counts = count_pretokens(input_path, special_tokens, num_processes)

        # 保存合并规则
        merges = []

        # 逐轮学习合并规则
        while len(vocab) < vocab_size:
            # 统计当前所有相邻token对的加权频次
            pair_counts = Counter()
            for tokens, frequency in pretoken_counts.items():
                for left, right in zip(tokens, tokens[1:]):
                    pair_counts[(left, right)] += frequency
            if not pair_counts:
                break

            # 优先选择频次更高或者字典序更大的
            best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            left, right = best_pair
            new_token = left + right
            vocab[len(vocab)] = new_token
            merges.append(best_pair)

            # 更新每种片段的token划分
            updated_counts = Counter()

            for tokens, frequency in pretoken_counts.items():
                updated_tokens = []
                index = 0

                while index < len(tokens):
                    if (
                        index + 1 < len(tokens)
                        and tokens[index] == left
                        and tokens[index + 1] == right
                    ):
                        updated_tokens.append(new_token)
                        index += 2
                    else:
                        updated_tokens.append(tokens[index])
                        index += 1

                updated_counts[tuple(updated_tokens)] += frequency
            pretoken_counts = updated_counts
        return cls(vocab, merges, special_tokens)
