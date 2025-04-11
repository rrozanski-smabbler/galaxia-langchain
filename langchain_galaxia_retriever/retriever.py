import time
import http.client
import json

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, model_validator, Field

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class GalaxiaClient:
    def __init__(self, api_url, api_key, knowledge_base_id, n_retries, wait_time):
        self.api_key = api_key
        self.api_url = api_url
        self.knowledge_base_id = knowledge_base_id
        self.n_retries = n_retries
        self.wait_time = wait_time

        self.headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}

    def initialize(
        self,
        conn: http.client.HTTPSConnection,
        question: str,
    ) -> dict:
        payload_0 = '{\n  "algorithmVersion":"%s",\n' % self.knowledge_base_id
        payload_1 = '  "text":"%s" \n}' % question.replace('"', '\\"')
        payload = payload_0 + payload_1
        conn.request("POST", "/analyze/initialize", payload, self.headers)
        res = conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))

    def check_status(
        self,
        conn: http.client.HTTPSConnection,
        init_res: dict,
    ) -> dict:
        payload = '{\n "operationId": "%s"\n}' % init_res["operationId"]
        conn.request("POST", "/analyze/status", payload, self.headers)

        res = conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))

    def get_result(self, conn: http.client.HTTPSConnection, init_res: dict) -> dict:
        payload = '{\n "operationId": "%s"\n}' % init_res["operationId"]
        conn.request("POST", "/analyze/result", payload, self.headers)

        res = conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))

    def retrieve(
        self,
        query: str,
    ) -> Union[dict, None]:
        conn = http.client.HTTPSConnection(self.api_url)

        flag_init = False
        for i in range(self.n_retries):
            init_res = self.initialize(conn, query)

            if "operationId" in init_res:
                flag_init = True
                break

            time.sleep(self.wait_time * i)

        if not flag_init:
            # failed to init
            return None

        flag_proc = False
        for i in range(1, self.n_retries + 1):
            time.sleep(self.wait_time * i)
            status = self.check_status(conn, init_res)

            if status["status"] == "processed":
                flag_proc = True
                break

        if flag_proc:
            res = self.get_result(conn, init_res)
            return res["result"]["resultItems"]

        else:
            # failed to process
            return None


class GalaxiaRetriever(BaseRetriever):
    """
    Galaxia Knowledge retriever.

    before using the API create your knowledge base here:
    beta.cloud.smabbler.com/

    learn more here:
    https://smabbler.gitbook.io/smabbler/api-rag/smabblers-api-rag

    Args:
        api_url : url of galaxia API, e.g. "beta.api.smabbler.com"
        api_key : API key
        knowledge_base_id : ID of the knowledge base (galaxia model)
        n_retries : the number of retires when calling the API
        wait_time : waiting time between attempts to call the API

    Example:
        .. code-block:: python

        from langchain_galaxia_retriever.retriever import GalaxiaRetriever

        gr = GalaxiaRetriever(
            api_url="beta.api.smabbler.com",
            api_key="<key>",
            knowledge_base_id="<knowledge_base_id>",
            n_retries=10,
            wait_time=5,
        )

        result = gr.invoke('<test question>')
        print(result)
    """
    api_url: str
    api_key: str
    knowledge_base_id: str
    client: Any
    n_retries: int = 10
    wait_time: int = 5

    @model_validator(mode="before")
    @classmethod
    def create_client(cls, values: Dict[str, Any]) -> Any:       
        if values.get("client") is not None:
            return values

        for f, def_v in zip(['n_retries', 'wait_time'], [5, 2]):
            if values.get(f) is None:
                values[f] = def_v
        
        client = GalaxiaClient(
            values['api_url'], 
            values['api_key'], 
            values['knowledge_base_id'], 
            values['n_retries'], 
            values['wait_time']
        )
 
        values["client"] = client
 
        return values
   

    def _get_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        response = self.client.retrieve(query.strip())
        documents = []

        if response is None:
            return documents

        for result in response:
            meta = {}
            meta['query'] = result['text']
            meta['model'] = result['model']
            meta['file'] = result['group']
            meta['score'] = result['rank']

            documents.append(
                Document(
                    page_content=result['category'],
                    metadata=meta,
                )
            )

        return documents
