import json

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0, max_tokens=1000)  # type: ignore


def parse_recipe(recipe_text: str) -> dict:
    # initial prompt to extract the instructions and ingredients
    # for ingredients extract quantity as well
    extract_prompt = PromptTemplate(
        input_variables=["recipe"],
        template="""{recipe}
        
        Extract the ingredients and instructions from the above, output as json with keys lowercase keys instructions and ingredients. For ingredients extract quantity if available. """,  # noqa
    )

    # secondary prompt to group extracted ingredients into categories
    grouping_prompt = PromptTemplate(
        input_variables=["ingredients"],
        template="""
        {ingredients}
        
        Group above ingredients into similar categories, output as json with lowercase keys categories:""",  # noqa
    )

    extract_chain = LLMChain(prompt=extract_prompt, llm=llm)
    grouping_chain = LLMChain(prompt=grouping_prompt, llm=llm)

    text_result = extract_chain.run(recipe_text)
    json_result = json.loads(text_result)

    ingredient_group_text = grouping_chain.run(json.dumps(json_result["ingredients"]))
    ingredient_group_json = json.loads(ingredient_group_text)

    json_result["grouped_ingredients"] = ingredient_group_json

    return json_result
