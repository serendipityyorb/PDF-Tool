from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def GetAIAnswer(model,api_key,base_url,pdf_file,memory,prompt):
    model = ChatOpenAI(model=model,api_key=api_key,base_url=base_url)
    #加载文档
    file_path = ".\local_file.pdf"
    pdf_file_content = pdf_file.read()
    with open(file_path,"wb") as fp:
        fp.write(pdf_file_content)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    #分割文档
    docs_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""]
    )
    docs_by_split = docs_splitter.split_documents(docs)

    #嵌套向量模型
    embedded_model = OpenAIEmbeddings(model="text-embedding-3-large",api_key=api_key,base_url=base_url,dimensions=1024)

#将嵌套向量模型和文本传入数据库
    db = FAISS.from_documents(docs_by_split,embedded_model)
    retriever = db.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(llm=model,memory=memory,retriever=retriever)
    res = chain.invoke({"question":prompt})
    return res

