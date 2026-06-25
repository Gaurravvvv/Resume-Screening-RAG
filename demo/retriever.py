import sys
sys.dont_write_bytecode = True

from typing import List
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate


RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
  """
  List of IDs of the applicants to retrieve resumes for
  """
  id_list: List[str] = Field(..., description="List of IDs of the applicants to retrieve resumes for")

class JobDescription(BaseModel):
  """
  Descriptions of a job to retrieve similar resumes for
  """
  job_description: str = Field(..., description="Descriptions of a job to retrieve similar resumes for") 



class RAGRetriever():
  def __init__(self, vectorstore_db, df):
    self.vectorstore = vectorstore_db
    self.df = df

  def __reciprocal_rank_fusion__(self, document_rank_list: list[dict], k=50):
    fused_scores = {}
    for doc_list in document_rank_list:
      for rank, (doc, _) in enumerate(doc_list.items()):
        if doc not in fused_scores:
          fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + k)
    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

  def __retrieve_docs_id__(self, question: str, k=50):
    docs_score = self.vectorstore.similarity_search_with_score(question, k=k)
    unique_docs = {}
    for doc, score in docs_score:
      doc_id = str(doc.metadata["ID"])
      if doc_id not in unique_docs or score < unique_docs[doc_id]:
        unique_docs[doc_id] = score
    return unique_docs

  def retrieve_id_and_rerank(self, subquestion_list: list):
    document_rank_list = []
    for subquestion in subquestion_list:
      document_rank_list.append(self.__retrieve_docs_id__(subquestion, k=20))
    reranked_documents = self.__reciprocal_rank_fusion__(document_rank_list)
    return reranked_documents

  def retrieve_documents_with_id(self, doc_id_with_score: dict, threshold=5):
    id_resume_dict = dict(zip(self.df["ID"].astype(str), self.df["Resume"]))
    retrieved_ids = list(sorted(doc_id_with_score, key=doc_id_with_score.get, reverse=True))[:threshold]
    retrieved_documents = []
    for id in retrieved_ids:
      if id in id_resume_dict:
        retrieved_documents.append("Applicant ID " + id + "\n" + id_resume_dict[id])
    return retrieved_documents 
   


class SelfQueryRetriever(RAGRetriever):
  def __init__(self, vectorstore_db, df):
    super().__init__(vectorstore_db, df)

    self.prompt = ChatPromptTemplate.from_messages([
      ("system", "You are an expert in talent acquisition."),
      ("user", "{input}")
    ])
    self.meta_data = {
      "rag_mode": "",
      "query_type": "no_retrieve",
      "extracted_input": "",
      "subquestion_list": [],
      "retrieved_docs_with_scores": []
    }

  def retrieve_docs(self, question: str, llm, rag_mode: str):
    self.meta_data = {
      "rag_mode": rag_mode,
      "query_type": "no_retrieve",
      "extracted_input": "",
      "subquestion_list": [],
      "retrieved_docs_with_scores": []
    }

    @tool(args_schema=ApplicantID)
    def retrieve_applicant_id(id_list: list):
      """Retrieve resumes for applicants in the id_list"""
      retrieved_resumes = []

      for id in id_list:
        try:
          matching_rows = self.df[self.df["ID"].astype(str) == id]
          if not matching_rows.empty:
            resume_df = matching_rows.iloc[0][["ID", "Resume"]]
            resume_with_id = "Applicant ID " + str(resume_df["ID"]) + "\n" + resume_df["Resume"]
            retrieved_resumes.append(resume_with_id)
          else:
            retrieved_resumes.append(f"Applicant ID {id}\n[Error: Resume not found in database]")
        except Exception as e:
          retrieved_resumes.append(f"Applicant ID {id}\n[Error retrieving resume: {str(e)}]")
      return retrieved_resumes

    @tool(args_schema=JobDescription)
    def retrieve_applicant_jd(job_description: str):
      """Retrieve similar resumes given a job description"""
      subquestion_list = [job_description]

      if rag_mode == "RAG Fusion":
        subquestion_list += llm.generate_subquestions(question)
        
      self.meta_data["subquestion_list"] = subquestion_list
      retrieved_ids = self.retrieve_id_and_rerank(subquestion_list)
      self.meta_data["retrieved_docs_with_scores"] = retrieved_ids
      retrieved_resumes = self.retrieve_documents_with_id(retrieved_ids)
      return retrieved_resumes
    
    self.meta_data["rag_mode"] = rag_mode
    llm_with_tools = llm.llm.bind_tools([retrieve_applicant_id, retrieve_applicant_jd])

    chain = self.prompt | llm_with_tools
    response = chain.invoke({"input": question})

    if response.tool_calls:
      tool_call = response.tool_calls[0]
      tool_name = tool_call["name"]
      tool_args = tool_call["args"]
      
      self.meta_data["query_type"] = tool_name
      self.meta_data["extracted_input"] = tool_args
      
      toolbox = {
        "retrieve_applicant_id": retrieve_applicant_id,
        "retrieve_applicant_jd": retrieve_applicant_jd
      }
      return toolbox[tool_name].run(tool_args)
    else:
      return response.content
