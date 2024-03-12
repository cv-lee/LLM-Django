import os
import json
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from .chat import VDB_PATH, EMBEDDING_MODEL


class Embedding:

    @staticmethod
    def embedding(
        document: Document, chunk_size: int = 500, chunk_overlap: int = 100
    ) -> None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "."],
        )

        document_chunk_list = text_splitter.split_documents([document])
        for i in reversed(range(len(document_chunk_list))):
            document_chunk = document_chunk_list[i]
            # document_chunk.page_content = (
            #     document_chunk.page_content
            #     + f"\n출처: {document_chunk.metadata['source'][5:-4]}"
            # )

            if "\udfb3" in document_chunk.page_content:
                print(
                    f"{document_chunk.metadata['source']} is deleted due to containing Non UTF-8 Character"
                )
                del document_chunk_list[i]

        chroma_db_dir = os.path.join(Path(__file__).parent, "chroma_db", VDB_PATH)
        chroma_db = Chroma.from_documents(
            document_chunk_list, EMBEDDING_MODEL, persist_directory=chroma_db_dir
        )


class DocumentPDF:
    def __init__(self, file_path: str) -> None:
        loader = PyPDFLoader(file_path)
        self.pages = loader.load()
        self.file_path: str = file_path
        self.start_page: int = 0
        self.end_page: int = len(self.pages)

    @staticmethod
    def clean_pages(page_list: List[Document]) -> List[Document]:
        for page in page_list:
            content = page.page_content
            lines = content.split("\n")
            header = lines[0]
            if "●●●" in header:
                clean_content = "\n".join(lines[1:])
                page.page_content = clean_content

        return page_list

    @staticmethod
    def combine_pages(page_list: List[Document]) -> Document:
        combined_page_content = ""

        for page in page_list:
            combined_page_content += page.page_content

        return Document(
            page_content=combined_page_content, metadata=page_list[0].metadata
        )

    def _preprocess(self) -> Document:
        clean_pages = self.clean_pages(self.pages)
        combine_pages = self.combine_pages(clean_pages)
        return combine_pages

    def embedding(self) -> None:
        Embedding.embedding(self._preprocess())


class DocumentHancom:
    pass


class DocumentWord:
    pass


if __name__ == "__main__":
    # Set document directory
    document_file_dir = os.path.join(Path(__file__).parent, "chroma_db/data_translated")

    # Make Chroma DB
    for doc_file_path in [
        os.path.join(document_file_dir, d) for d in os.listdir(document_file_dir)
    ]:
        _, file_ext = os.path.splitext(doc_file_path)
        if file_ext == ".pdf":
            DocumentPDF(doc_file_path).embedding()
        elif file_ext == ".doc" or file_ext == ".docx":
            raise NotImplementedError()
        elif file_ext == ".hwp" or file_ext == ".hwpx":
            raise NotImplementedError()
        else:
            raise ValueError(f"{file_ext} is not supported File Extensions!")
