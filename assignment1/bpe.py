from collections import Counter
import regex
import os
from typing import BinaryIO
from concurrent.futures import ProcessPoolExecutor


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件切分成可以独立统计的多个部分。
    如果不同分块的边界重合，最终返回的分块数量可能少于期望值。
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


# 预分词规则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 去掉重复的特殊token并保留顺序
    special_tokens = list(dict.fromkeys(special_tokens))

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
    return vocab, merges
