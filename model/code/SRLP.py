import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# 加载 LTP
from ltp import LTP

# BERT模型
BERT_MODEL_PATH = 'bert-base-chinese'

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
model = BertModel.from_pretrained(BERT_MODEL_PATH)

#file_path = "C:/Users/is_li/Desktop/paper/github/stock on MLOps/uploads"
#tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH,do_lower_case=True)
#PRE_TRAINED_MODEL_NAME = file_path +'/model/NLP_Astock/ROBERT_4_model.bin'


# 加载 LTP
ltp = LTP()

# 函数：计算单个句子的 V、A0、A1 向量表示
def sentence_vectors(sentence):
    seg_result = ltp([sentence])
    hidden = ltp([seg_result])
    srls = ltp([hidden])
    V, A0, A1 = [], [], []

    for srl in srls[0]:
        if srl[0] == "A0":
            A0.append(seg_result.words[0][srl[1]])
        elif srl[0] == "A1":
            A1.append(seg_result.words[0][srl[1]])
        elif srl[0] == "V":
            V.append(seg_result.words[0][srl[1]])

    # 使用 BERT 计算句子的向量表示
    tokenized_sentence = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokenized_sentence)
        hidden_states = outputs.last_hidden_state.numpy()[0]

    # 池化 V、A0、A1 的向量表示
    def pool_embeddings(indices):
        return np.mean(hidden_states[indices], axis=0)

    V_indices = [i for i, token in enumerate(tokenized_sentence['input_ids'][0]) if token.item() in [tokenizer.convert_tokens_to_ids(t) for t in V]]
    A0_indices = [i for i, token in enumerate(tokenized_sentence['input_ids'][0]) if token.item() in [tokenizer.convert_tokens_to_ids(t) for t in A0]]
    A1_indices = [i for i, token in enumerate(tokenized_sentence['input_ids'][0]) if token.item() in [tokenizer.convert_tokens_to_ids(t) for t in A1]]

    V_vector = pool_embeddings(V_indices)
    A0_vector = pool_embeddings(A0_indices)
    A1_vector = pool_embeddings(A1_indices)

    return V_vector, A0_vector, A1_vector

# 句子列表
sentences = [
    '科锐国际股东CareerHK减持公司股份199万股',
    '某公司高管决定增持本公司股票',
    '投资者对明天的股票行情持观望态度'
]

# 调用函数计算句子的 V、A0、A1 向量表示
for sentence in sentences:
    V_vector, A0_vector, A1_vector = sentence_vectors(sentence)
    print(f"Sentence: {sentence}")
    print(f"V_vector: {V_vector}")
    print(f"A0_vector: {A0_vector}")
    print(f"A1_vector: {A1_vector}")
    print("\n")
