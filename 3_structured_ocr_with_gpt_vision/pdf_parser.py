import base64
import os
from typing import Optional

import requests
from pydantic import BaseModel


class Section(BaseModel):
    title: str
    section: str
    text: str


class Connector(BaseModel):
    target_node_id: int
    condition: Optional[str] = None


class Node(BaseModel):
    id: int
    description: str
    connectors: list[Connector] = []


class Flowchart(BaseModel):
    nodes: list[Node]


class Figure(BaseModel):
    title: str
    figure_subtitle: str
    figure_number: int
    figure_data: Flowchart


class Page(BaseModel):
    title: str
    page_number: int
    sections: list[Section] = []
    figures: list[Figure] = []


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def prepare_payload(base64_image):
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        You are an expert in PDFs. You are helping a user extract text from a PDF.
                        Extract the text from the image as a structured json output.
                        Extract the data using the following schema:
                        {Page.model_json_schema()}
                        Example:
                        {{
                            "title": "Title",
                            "page_number": 1,
                            "sections": [
                            ...
                            ],
                            "figures": [
                            ...
                            ]
                        }}
                        """,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        Please extract the text from the image as a structured json output.
                        Include a "figure" key for any flowchart or diagram.
                        """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        "max_tokens": 1000,
    }


def make_request(headers, payload):
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    image_path = "data/gpt4v-ocr.png"
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = prepare_payload(base64_image)
    response = make_request(headers, payload)

    if response:
        print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
