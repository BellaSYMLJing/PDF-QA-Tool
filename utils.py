from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def qa_agent(api_key,memory,uploaded_file,question):
    model = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=api_key,openai_api_base = "https://api.aigc369.com/v1")

    # 把外部文档加载进来
    # read方法返回的是内容的二进制数据bytes
    file_content = uploaded_file.read()
    # 新建一个用于储存PDF内容的临时文件路径
    temp_file_path = "temp.pdf"
    # 把读取的二进制内容写入，写入二进制的模式是"wb"
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # 文本切割成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 50,
        separators = ["\n","。","！","？","，","、",""]
    )
    texts = text_splitter.split_documents(docs)

    # 嵌入向量
    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key,openai_api_base = "https://api.aigc369.com/v1")

    # 储存进向量数据库
    db = FAISS.from_documents(texts,embeddings_model)

    # 得到检索器，可以在数据库中检索
    retriever = db.as_retriever()

    # 带记忆的检索增强链
    qa = ConversationalRetrievalChain.from_llm(
        llm = model,
        retriever = retriever,
        memory = memory
    )
    response = qa.invoke({"chat_history":memory,"question":question})
    return response