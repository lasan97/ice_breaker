from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import os

from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()

    print("Hello Langchain!")

    summery_template = """
    생성하려는 사람에 대한 링크드인 정보 {information}가 주어집니다.
    1. 짧은 요약
    2. 그들에 대한 두 가지 흥미로운 사실
    """

    summery_prompt_template = PromptTemplate(input_variables=["information"], template=summery_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summery_prompt_template)

    linkedin_data = scrape_linkedin_profile("", mock=True)

    res = chain.invoke(input={"information": linkedin_data})

    print(res)
    print(res['text'])
