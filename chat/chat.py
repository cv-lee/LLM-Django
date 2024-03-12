import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .local_llm import LocalLLM
from .translate import Translator

EMBEDDING_MODEL = GPT4AllEmbeddings()
LLM = LocalLLM().llm
TRANSLATOR = Translator()

TOP_K = 10
VDB_PATH = "vdb_20240227"

def ask(question, llm=LLM, embedding_model=EMBEDDING_MODEL):
    chroma_db_dir = os.path.join(Path(__file__).parent, "chroma_db", VDB_PATH)
    chroma_db = Chroma(
        persist_directory=chroma_db_dir, embedding_function=embedding_model
    )
    retriever = chroma_db.as_retriever(search_kwargs={"k": TOP_K})

    prompt_template = PromptTemplate.from_template(
        """ 
        Answer the question at the end in detail using the following context.
        If you don't find anything relevant to the question in context, answer "관련 정보가 없습니다." If you do, answer in as much detail as possible.
        Even if you don't have an exact answer, try to find something that is close enough.
        Prioritize laws and regulations among your sources, and refer to other sources if there are no laws and regulations.
        However, if you are asked to make any judgment, please answer "I cannot make a judgment". The responses must be in Korean.
         Context: {context}
         Question: {question}
         Answer:
     """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )

    question = TRANSLATOR.translate(question, "EN")
    print(question)
    try:
        answer = qa_chain.invoke({"query": question})
    except Exception as e:
        answer = {
            "result": "관련 정보가 없습니다.",
            "source_documents": [],
        }
        print(f"An error occurred: {e}")
    print(answer["result"])
    result = TRANSLATOR.translate(answer["result"], "KO")
    refs = []
    for src in answer['source_documents']:
        src_doc = src.metadata['source'].split("/")[-1].split(".pdf")[0]
        if src_doc not in refs:
            refs += [src_doc]
    result += "\n출처: "+"\n".join(refs)
    result = result.replace('\n', '<br>')
    print(result)
    return result
    
    