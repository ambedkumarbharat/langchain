from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

docs = """
        A large language model (LLM) is an AI model (typically a neural network) trained on a vast amount of text for natural language processing tasks, especially language generation. LLMs can typically generate, summarize, translate, and analyze text in many contexts.[1] They are the basis for many modern chatbots, such as ChatGPT, Claude, Gemini, Grok, and DeepSeek.

        LLMs are typically based on transformer architecture.[2] Generative pre-trained transformers (GPTs) are a type of LLM that is pre-trained to predict the next word.[3] GPTs are then often fine-tuned to follow instructions and to behave as assistants.[4]

        Biased or inaccurate training data can make an LLM's output less reliable. Benchmark evaluations for LLMs attempt to measure model reasoning, factual accuracy, alignment, and safety.

        Dataset preprocessing
        See also: List of datasets for machine-learning research § Internet
        Tokenization
        As machine learning algorithms process numbers rather than text, the text must be converted to numbers. In the first step, a vocabulary is decided upon, then integer indices are arbitrarily but uniquely assigned to each vocabulary entry, and finally, an embedding is associated with the integer index. Algorithms include byte-pair encoding (BPE) and WordPiece. There are also special tokens serving as control characters, such as [MASK] for masked-out token (as used in BERT), and [UNK] ("unknown") for characters not appearing in the vocabulary. Also, some special symbols are used to denote special text formatting. For example, "Ġ" denotes a preceding whitespace in RoBERTa and GPT and "##" denotes continuation of a preceding word in BERT.[23]

        For example, the BPE tokenizer used by the legacy version of GPT-3 would split tokenizer: texts -> series of numerical "tokens" as

        token	izer	:	 texts	 ->	series	 of	 numerical	 "	t	ok	ens	"
        Tokenization also compresses the datasets. Because LLMs generally require input to be an array that is not jagged, the shorter texts must be "padded" until they match the length of the longest one.

        Byte-pair encoding
        Main article: Byte-pair encoding
        As an example, consider a tokenizer based on byte-pair encoding. In the first step, all unique characters (including blanks and punctuation marks) are treated as an initial set of n-grams (i.e. initial set of uni-grams). Successively the most frequent pair of adjacent characters is merged into a bi-gram and all instances of the pair are replaced by it. All occurrences of adjacent pairs of (previously merged) n-grams that most frequently occur together are then again merged into even lengthier n-gram, until a vocabulary of prescribed size is obtained. After a tokenizer is trained, any text can be tokenized by it, as long as it does not contain characters not appearing in the initial-set of uni-grams.[24]

        Dataset cleaning
        Main article: Data cleansing
        In the context of training LLMs, datasets are typically cleaned by removing low-quality, duplicated, or toxic data.[25] Cleaned datasets can increase training efficiency and lead to improved downstream performance.[26] A trained LLM can be used to clean datasets for training a further LLM.[27]

        With the increasing proportion of LLM-generated content on the web, data cleaning in the future may include filtering out such content. LLM-generated content can pose a problem if the content is similar to human text (making filtering difficult) but of lower quality (degrading performance of models trained on it).[1]

        Synthetic data
        Main article: Synthetic data
        Training of largest language models might need more linguistic data than naturally available, or that the naturally occurring data is of insufficient quality. In these cases, synthetic data might be used.
"""

prompt1 = PromptTemplate(
    template="generate best notes from text document \n notes -> {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="generate best quiz question with options from text document \n quiz -> {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz in single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

openrouter_llm = ChatOpenRouter(model="nex-agi/nex-n2.5-mini:free")

groq_llm = ChatGroq(model="openai/gpt-oss-120b", temperature= 0.7)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | groq_llm | parser,
    'quiz': prompt2 | openrouter_llm | parser
})

chain = prompt3 | groq_llm | parser

merge_chain = parallel_chain | chain

result = merge_chain.invoke({"text":docs})

print(result)

merge_chain.get_graph().print_ascii()